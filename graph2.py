from typing import Tuple, List, Optional
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from yfiles_jupyter_graphs import GraphWidget
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
import os
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter

from dotenv import load_dotenv

st.set_page_config(layout="wide")
st.subheader("APCO Insight - GenAI | Knowledge Graph | LLM")
#st.markdown("---")
st.sidebar.title("APCO Insight")
st.sidebar.markdown("---")
st.sidebar.write('This application enables the creation of a knowledge graph from any text data, allows you to add new data to an existing Neo4j knowledge graph, and provides the ability to search queries within the graph.')
tab1, tab2, tab3 = st.tabs([" APCO QNA", "Knowledge Graph", "Add New Data"])


with tab1: 
    graph = Neo4jGraph()

    # NEO4J_URI="neo4j+s://bcd2c1de.databases.neo4j.io"
    # NEO4J_USERNAME="neo4j"
    # NEO4J_PASSWORD="aex5F0a3zASVNRUUM2uF-Cdw03W3rb_9DnBY2aNy42g"
    # OPENAI_API_KEY='sk-QM07AW9xy_FWac5g3K-MtRzy58uRgGO8cZFjOoHvAjT3BlbkFJdGCyPLP_SFl3pZz3mHHJiEY-a_9Lq7Qfxl-DDrd0wA'

    NEO4J_URI= os.getenv("NEO4J_URI")
    NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["NEO4J_URI"] = NEO4J_URI
    os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
    os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
    
    
    from langchain_openai import ChatOpenAI
    llm=ChatOpenAI(temperature=0, model_name="gpt-4o")
    
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    
    graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
    
    
    # Extract entities from text
    class Entities(BaseModel):
        """Identifying information about entities."""
    
        names: List[str] = Field(
            ...,
            description="All the person, organization, or business entities that "
            "appear in the text",
        )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )
    
    entity_chain = prompt | llm.with_structured_output(Entities)
    
    def generate_full_text_query(input: str) -> str:
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()
    
    # Fulltext index query
    def structured_retriever(question: str) -> str:
        result = ""
        entities = entity_chain.invoke({"question": question})
        for entity in entities.names:
            response = graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                  WITH node
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result
    
    def retriever(question: str):
        print(f"Search query: {question}")
        structured_data = structured_retriever(question)
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        final_data = f"""Structured data:
    {structured_data}
    Unstructured data:
    {"#Document ". join(unstructured_data)}
        """
        return final_data
    
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    
    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer
    
    _search_query = RunnableBranch(
        # If input includes chat_history, we condense it with the follow-up question
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),  # Condense follow-up question and chat into a standalone_question
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0)
            | StrOutputParser(),
        ),
        # Else, we have no chat history, so just pass through the question
        RunnableLambda(lambda x : x["question"]),
    )
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    Use natural language and be concise.
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        RunnableParallel(
            {
                "context": _search_query | retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    #x = chain.invoke({"question": "what was problem with Maggie and did APCO helped?"})
    
    import streamlit as st
    #st.title(x)
    # User input
    question = st.text_input("Ask a question:", "")
    if st.button("Submit"):
        if question:
            with st.spinner("Processing..."):
                x = chain.invoke({"question": question})
                st.write(f"Answer: {x}")
        else:
            st.error("Please enter a question!")

#############
# Function to fetch graph data from Neo4j
def fetch_graph_data(cypher_query: str):
    driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    session = driver.session()
    
    result = session.run(cypher_query)
    nodes = []
    edges = []

    for record in result:
        start_node = record['s']
        end_node = record['t']
        relationship = record['r']

        # Create nodes and edges in a simple format
        nodes.append({"id": start_node.id, "label": start_node["id"], "group": "start"})
        nodes.append({"id": end_node.id, "label": end_node["id"], "group": "end"})
        edges.append({"from": start_node.id, "to": end_node.id, "label": relationship.type})
    
    # Remove duplicate nodes
    nodes = [dict(t) for t in {tuple(d.items()) for d in nodes}]
    
    return {"nodes": nodes, "edges": edges}

# Function to create and display graph using PyVis
def show_graph(nodes, edges):
    net = Network(height="1200px", width="1800px", bgcolor='#222222', font_color='white')
    
    # Add nodes and edges to the graph
    for node in nodes:
        net.add_node(node["id"], label=node["label"], title=node["label"], group=node["group"])

    for edge in edges:
        net.add_edge(edge["from"], edge["to"], title=edge["label"], label=edge["label"])
    
    # Customize graph options for improved aesthetics and interactivity
    net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 2,
        "size": 15,
        "color": {
          "border": "#6AA84F",
          "background": "#1C8A5F",
          "highlight": {
            "border": "#F39C12",
            "background": "#E67E22"
          }
        },
        "font": {
          "color": "#ffffff",
          "size": 10
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "width": 0.5,
        "size": 8,
        "smooth": {
          "type": "cubicBezier",
          "forceDirection": "none",
          "roundness": 0.5
        },
        "font": {
          
          "size": 6.5
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": true
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -20000,
          "centralGravity": 0.3,
          "springLength": 120,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "minVelocity": 0.75
      }
    }
    """)
    



    # Generate the graph in HTML format and save it
    net.save_graph("graph.html")

    # Display the graph in Streamlit
    HtmlFile = open("graph.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=900)

# Streamlit app
def kbase():
    #st.title("APCO Knowledge Graph/LLM Application")
    #st.markdown('---')
    #st.subheader("1. APCO Knowledge graph")

    # Define the default Cypher query
    default_cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"
    
    # Text area for user to input their Cypher query
    cypher_query = st.text_area("Enter a Cypher Query", default_cypher)

    # Button to fetch and display the graph
    if st.button("Run Query and Show Graph"):
        # Fetch data from Neo4j
        graph_data = fetch_graph_data(cypher_query)

        # Display graph using PyVis
        show_graph(graph_data["nodes"], graph_data["edges"])



    # Initialize the Neo4j Graph
    graph = Neo4jGraph()




##############



with tab2:
    kbase()


with tab3:
    # Collect user input
    title = st.text_input("Enter Title")
    summary = st.text_area("Enter Summary")
    source = st.text_input("Enter Source URL")
    content = st.text_area("Enter Content/Text")
    
    if st.button("Submit and Generate Graph"):
        # Prepare metadata and create a Document instance
        if title and summary and source and content:
            document = {
                'metadata': {
                    'title': title,
                    'summary': summary,
                    'source': source
                },
                'page_content': content
            }
    
            st.write(f"Document Created: {document}")
            
            # Create the Document for processing
            doc = Document(page_content=content)
            
            # Split the document
            text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
            documents = text_splitter.split_documents([doc])
    
            # Use the LLM to transform the document into a graph format
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
            llm_transformer = LLMGraphTransformer(llm=llm)
            graph_documents = llm_transformer.convert_to_graph_documents(documents)
    
            # Add the graph to Neo4j
            graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )
    
            st.success("Graph Generated and Added to Database")
            st.write("Graph Representation:")
            st.json(graph_documents)
    
        else:
            st.warning("Please fill in all the fields.")
    
