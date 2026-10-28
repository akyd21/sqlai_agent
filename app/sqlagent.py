import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import URL

load_dotenv()


SYSTEM_PROMPT = """
You are a task management assistant that interacts with a SQL database containing a
'tasks' table.

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC.
2. After CREATE/UPDATE, confirm with a SELECT query.
3. If the user requests a list of tasks, present the output in a structured table format.

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""


@st.cache_resource
def create_agent_resources():
    mysql_password = os.getenv("MYSQL_PASSWORD")
    if not mysql_password:
        raise RuntimeError("MYSQL_PASSWORD is missing from the .env file.")

    database_url = URL.create(
        "mysql+pymysql",
        username=os.getenv("MYSQL_USER", "root"),
        password=mysql_password,
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.getenv("MYSQL_DATABASE", "task_manager"),
    )
    db = SQLDatabase.from_uri(database_url)

    db.run("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTO_INCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        status ENUM('pending', 'in_progress', 'completed') NOT NULL DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    model = ChatGroq(model="openai/gpt-oss-20b", reasoning_format="hidden")
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    agent = create_agent(
        model=model,
        tools=toolkit.get_tools(),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    return agent


st.set_page_config(page_title="SQL Task Agent", page_icon=":bar_chart:")
st.title("SQL Task Agent")
st.caption("Manage tasks stored in MySQL using natural language.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

try:
    agent = create_agent_resources()
except Exception as error:
    st.error(f"Could not connect to the database or initialize the agent: {error}")
    st.stop()

if prompt := st.chat_input("Ask me to create, read, update, or delete a task..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            try:
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    {"configurable": {"thread_id": "streamlit-session"}},
                )
                answer = response["messages"][-1].content
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as error:
                st.error(f"Request failed: {error}")
