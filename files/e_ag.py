from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage,AnyMessage,ToolMessage,AIMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import os
from pydantic import BaseModel,Field
import gradio as gr
from typing import TypedDict, Annotated
import operator

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_HANDLER_TO_CHARS"] = "false"
memory = MemorySaver()


class InputCal(BaseModel):
    operation: str = Field(description="full operation with numbers like 2*9")

@tool(args_schema=InputCal)
def calculate(operation: str) -> str:
    """perform mathematical calculations like addition ,multiplication,division and else"""
    try:
        return str(eval(operation))
    except Exception as e:
        return f"error :{str(e)}"

@tool
def search_web(query: str) -> str:
    """search the internet for current information about any topic"""
    try:
        response = tavily.search(
            query=query,
            search_depth="basic",
            max_results=2,
            include_answer=True
        )
        if response["answer"]:
            return response["answer"]
        else:
            return "no results found"
    except Exception as e:
        return f"search error:{str(e)}"


import requests
@tool
def get_weather(city: str) -> str:
    """
    الحصول على الطقس من wttr.in مباشرة
    """
    try:
        city = city.strip().lower()
        
        # wttr.in يعمل مباشرة مع اسم المدينة
        url = f"https://wttr.in/{city}?format=%C+%t+%w+%h"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.text.strip().split()
        
        if len(data) >= 4:
            condition = " ".join(data[:-3])
            temp = data[-3]
            wind = data[-2]
            humidity = data[-1]
            # return response.text
            return f"""🌤️ **حالة الطقس في {city.title()}**

☁️ **الحالة:** {condition}
🌡️ **درجة الحرارة:** {temp}
💨 **الرياح:** {wind}
💧 **الرطوبة:** {humidity}
📍 **المصدر:** wttr.in"""
        else:
            return f"❌ لم يتم العثور على مدينة {city}"
            
    except Exception as e:
        return f"❌ خطأ: {str(e)}"
    


@tool
def essay_writer(task: str, max_revisions: int = 1) -> str:
    """call these tool to write essay about the task with many revisions you can sellect max revisions"""
    return f"تتم كتابة مقال عن {task} بعدد مراجعات {max_revisions}"

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
agent_model = model.bind_tools([calculate, search_web, essay_writer])

# ✅ إصلاح 1: t.name بدل t["name"]
tools = {t.name: t for t in [calculate, search_web, essay_writer,get_weather]}

PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections.'write with user language'"""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 
------
{content}'write with user language'"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc.'write with user language'"""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max.'write with user language'"""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can be used when making any requested revisions (as outlined below). Generate a list of search queries that will gather any relevant information. Only generate 3 queries max.

CRITICAL INSTRUCTION: You must output the response using the exact language that the user is currently using to communicate with you. Even if the search queries or results are in English or any other language, your final output must be in the user's language. Do not translate the user's intent; respond natively in their language.'"""

class Queries(BaseModel):
    queries: List[str]

def agent(state: AgentState):
    messages = [SystemMessage(content="you are a helpful assistant")] + state["messages"]
    response = agent_model.invoke(messages)
    return {'messages': [response]}

def is_tools_con(state: AgentState):
    return bool(state['messages'][-1].tool_calls)

def essay_start(state: AgentState):
    a = {}
    for message in reversed(state['messages']):
        if isinstance(message, AIMessage):
            for c in message.tool_calls:
                if c['name'] == "essay_writer":
                    task = c['args']['task']
                    max_revisions = c['args'].get('max_revisions', 1)  # ✅ يأخذ 1 افتراضياً              
                    a = {
                        'task': task,
                        "max_revisions": max_revisions,
                        "revision_number": 1,
                        "content": [],
                        "plan": "",
                        "draft": "",
                        "critique": ""
                    }
                    break
            break
    mes = [SystemMessage(content="you are a helpful assistant, if the user ask about an essay make your response be only what the tool return 'only one line of information'")] + state["messages"]

    response = agent_model.invoke(mes)
    messages = {"messages": [response]}
    messages.update(a)
    return messages

def is_essay_con(state: AgentState):
    for message in reversed(state['messages']):
        if isinstance(message, AIMessage):
            for c in message.tool_calls:
                if c['name'] == "essay_writer":
                    return True
            break
    return False

def do_action(state: AgentState):
    tool_calls = state["messages"][-1].tool_calls
    tools_res = []
    for t in tool_calls:
        if t["name"] in tools:
            res = tools[t["name"]].invoke(t["args"])
            tools_res.append(
                ToolMessage(
                    tool_call_id=t["id"],
                    name=t["name"],
                    content=str(res)
                )
            )
        else:
            tools_res.append(
                ToolMessage(
                    tool_call_id=t["id"],
                    name=t["name"],
                    content=f"error not found tool {t['name']}"
                )
            )
    return {"messages": tools_res}

def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}

def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries, method="json_schema").invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries, method="json_schema").invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

# ============================================================
# بناء الغراف
# ============================================================
builder = StateGraph(AgentState)
builder.add_node("llm", agent)
builder.add_node("action", do_action)
builder.add_node("planner", plan_node)
builder.add_node("essays", essay_start)
builder.add_node("research_plan", research_plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("llm")
builder.add_edge("essays", "planner")
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")
builder.add_conditional_edges("generate", should_continue, {END: END, "reflect": "reflect"})
builder.add_conditional_edges("llm", is_tools_con, {True: "action", False: END})
builder.add_conditional_edges("action", is_essay_con, {True: "essays", False: "llm"})

graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["planner", "research_plan", "generate", "reflect", "research_critique"]
)

# ============================================================
# متغيرات الجلسة
# ============================================================
# ✅ إصلاح 2: إزالة تعريف thread المكرر — تعريف واحد فقط
thread = {"configurable": {"thread_id": "user_0"}}
thread_count = 0
accumulated_text = ""

node_names = {
    "planner": "📋 المخطط",
    "research_plan": "🔍 الباحث الأول",
    "generate": "✍️ الكاتب",
    "reflect": "🔎 الناقد",
    "research_critique": "🔍 الباحث الثاني"
}

ESSAY_NODES = {"planner", "research_plan", "generate", "reflect", "research_critique"}

# ============================================================
# دوال الواجهة
# ============================================================
def new_chat():
    global thread_count
    thread_count += 1
    return "", [], gr.update(visible=False, value=""), gr.update(visible=False,value="اكمل")

async def chat_gui(message, history):
    global accumulated_text

    current_thread = {"configurable": {"thread_id": f"user_{str(thread_count)}"}}

    if not message or not message.strip():
        yield gr.update(placeholder="الرجاء اكتب السؤال"), history, gr.update(), gr.update()
        return

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    full = ""
    essay_requested = False

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=message)]},
        current_thread,
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            node = event.get("metadata", {}).get("langgraph_node", "")
            chunk = event["data"]["chunk"].content
            if not chunk:
                continue
            if node in ("llm", "essays"):
                full += chunk
                history[-1]["content"] = full
                yield "", history, gr.update(), gr.update()

        elif event["event"] == "on_tool_start" and event["name"] == "essay_writer":
            essay_requested = True
            task_topic = event["data"].get("input", {}).get("task", message)
            accumulated_text = f"\n\n# مقال عن {task_topic}\n\n"

    if essay_requested:
        yield "", history, gr.update(visible=True, value=accumulated_text), gr.update(visible=True)
    else:
        yield "", history, gr.update(visible=False, value=""), gr.update(visible=False)

async def continue_graph():
    global accumulated_text

    state = graph.get_state(thread)
    current_node = state.next[0] if state.next else None

    if not current_node:
        accumulated_text += "\n\n✅ **اكتمل المقال!**"
        yield accumulated_text, gr.update(visible=False)
        return

    accumulated_text += f"\n\n---\n### {node_names.get(current_node, current_node)}\n"

    async for event in graph.astream_events(None, thread, version="v2"):
        if event["event"] == "on_chat_model_stream":
            node = event.get("metadata", {}).get("langgraph_node", "")
            if node in ESSAY_NODES:
                chunk = event["data"]["chunk"].content
                if chunk:
                    accumulated_text += chunk
                    yield accumulated_text, gr.update(visible=True)

# ============================================================
# بناء الواجهة
# ============================================================
with gr.Blocks() as demo:
    gr.Markdown("# مساعد ذكي 🤖")

    chatbot = gr.Chatbot(height=200)
    msg = gr.Textbox(placeholder="اكتب رسالتك هنا من فضلك")

    with gr.Row():
        sendbutton = gr.Button("إرسال", variant="primary")
        addbutton = gr.Button("محادثة جديدة", variant="secondary")

    out = gr.Markdown(height=300, visible=False)
    continue_btn = gr.Button("▶️ أكمل", variant="secondary", visible=False)

    msg.submit(chat_gui, [msg, chatbot], [msg, chatbot, out, continue_btn])
    sendbutton.click(chat_gui, [msg, chatbot], [msg, chatbot, out, continue_btn])
    addbutton.click(new_chat, None, [msg, chatbot, out, continue_btn])
    continue_btn.click(continue_graph, inputs=[], outputs=[out, continue_btn])

demo.launch(share=True)

