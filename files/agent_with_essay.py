from dotenv import load_dotenv
load_dotenv()
from tavily import TavilyClient
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import os
from pydantic import BaseModel
import gradio as gr
from langchain.tools import tool

memory = MemorySaver()

class EssayState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

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

def plan_node(state: EssayState):
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}

def research_plan_node(state: EssayState):
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

def generation_node(state: EssayState):
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

def reflection_node(state: EssayState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

def research_critique_node(state: EssayState):
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
# بناء الجراف
# ============================================================
builder = StateGraph(EssayState)

builder.add_node("planner", plan_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")
builder.add_conditional_edges(
    "generate",
    should_continue,
    {END: END, "reflect": "reflect"}
)

graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["planner","research_plan", "generate", "reflect", "research_critique"]
)

# ============================================================
# متغيرات الجلسة
# ============================================================
thread = {"configurable": {"thread_id": "1"}}
node_names = {
    "planner": "📋 المخطط",
    "research_plan": "🔍 الباحث الأول",
    "generate": "✍️ الكاتب",
    "reflect": "🔎 الناقد",
    "research_critique": "🔍 الباحث الثاني"
}


@tool
def essay_writer(task: str, max_revisions: int = 1) -> str:
    """call these tool to write essay about the task with many revisions you can sellect max revisions"""
    

    return f"تابع تقدم المقال {task} في القسم السفلي بعدد مراجعات {max_revisions}"
memorya=MemorySaver()

class InputCal(BaseModel):
    operation: str =Field(description="full operation with numbers like 2*9")

@tool(args_schema=InputCal)
def calculate(operation:str)->str:
    """perform mathematical calculations like addition ,multiplication,division and else"""
    try :
        return str(eval(operation))
    except Exception as e:
        return f"error :{str(e)}"
    
@tool
def search_web(query:str)->str:
    """search the internet for current information about any topic"""
    try:
        response=tavily.search(
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
     
tools=[calculate,search_web,essay_writer]

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage],operator.add]


class Agent:
    def __init__(self,model ,tools,checkpointer,system=""):
        self.system=system
        self.tools={t.name :t for t in tools}
        self.model=model.bind_tools(tools)
        graph=StateGraph(AgentState)
        graph.add_node("llm",self.model_call)
        graph.add_node("action",self.do_action)
        graph.add_conditional_edges("llm",self.is_action,{True:"action",False:END})
        graph.add_edge("action","llm")
        graph.set_entry_point("llm")
        self.graph=graph.compile(checkpointer=checkpointer)
    def model_call(self,state:AgentState):
        messages=state["messages"]
        if self.system:
            messages=[SystemMessage(content=self.system)]+messages
        response=self.model.invoke(messages)
        return {"messages":[response]}
    def is_action(self,state:AgentState):
        tool_calls=state["messages"][-1].tool_calls
        return bool(tool_calls)
    def do_action(self,state:AgentState):
        tool_calls=state["messages"][-1].tool_calls
        tools_res=[]
        for t in tool_calls:
            if t["name"] in self.tools:
                res=self.tools[t["name"]].invoke(t["args"])
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
                        content=f"error not found tool{t['name']}"
                    )
                )            
        return {"messages":tools_res}


model2=ChatOpenAI(temperature=0)
thread_count=0

def new_chat():
    global thread_count
    thread_count+=1
    return "",[], gr.update(visible=False, value=""), gr.update(visible=False, value="▶️ أكمل")
sey_m='''You are a highly efficient assistant equipped with multiple tools. You must strictly follow these instructions:

1. LANGUAGE RECOGNITION: Always detect the language the user is communicating with and respond natively in that exact same language. Never switch to English unless explicitly requested by the user.

2. TOOL INVOCATION DETERMINATION: Evaluate the user's input carefully. If the request requires any mathematical calculations, use the 'calculate' tool. If it requires current real-time or general information, use the 'search_web' tool. If the user explicitly asks to write, generate, or review an essay (مقال), you MUST immediately invoke the 'essay_writer' tool.

3. CRITICAL INSTRUCTION FOR ESSAY WRITER TOOL: When you decide to call the 'essay_writer' tool, your final output response to the user MUST ONLY contain the exact text returned by the tool's execution. DO NOT add any greetings, apologies, introductions, conclusions, conversational padding, or extra explanations (e.g., Do not say: "Sure, I will call the tool for you" or "Here is the result:"). Output the tool's string return and absolutely nothing else.'''
abot=Agent(model=model2,tools=tools,checkpointer=memorya,system=sey_m)
async def chat_gui(message, history):
    global accumulated_text
    thread_agent = {"configurable": {"thread_id": f"user_{str(thread_count)}"}}
    if not message or not message.strip():
        yield gr.update(placeholder="الرجاء اكتب السؤال"), history, gr.update(), gr.update()
        return
        
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    full = ""
    
    # 1. تجهيز متغيرات افتراضية خارج الحلقة
    essay_requested = False
    task_topic = message       # قيمة احتياطية في حال لم نجد المدخلات
    max_revs = 1               # قيمة افتراضية للمراجعات
    
    # البث الطبيعي للوكيل الأب
    async for event in abot.graph.astream_events({"messages": [HumanMessage(content=message)]}, thread_agent, version="v1"):
        if event["event"] == "on_chat_model_stream":
            chunk = event['data']["chunk"].content
            if chunk:
                full += chunk
                history[-1]["content"] = full
                yield "", history, gr.update(), gr.update()
        
        # 2. السحر هنا: التقاط لحظة تشغيل الأداة واستخراج مدخلاتها بدقة
        elif event["event"] == "on_tool_start" and event["name"] == "essay_writer":
            essay_requested = True
            
            # استخراج قاموس المدخلات (Arguments) التي صاغها الذكاء الاصطناعي للأداة
            tool_inputs = event["data"].get("input", {})
            
            # سحب القيم وتخزينها في المتغيرات التي جهزناها في الأعلى
            task_topic = tool_inputs.get("task", message)
            max_revs = tool_inputs.get("max_revisions", 1)

    # 3. الآن المتغيرات أصبحت معرّفة وجاهزة للاستخدام بأمان خارج حلقة البث!
    if essay_requested:
        initial_state = {
            'task': task_topic,        # تم التقاطه من الأداة بنجاح
            "max_revisions": max_revs,  # تم التقاطه من الأداة بنجاح
            "revision_number": 1,
            "content": [],
            "plan": "",
            "draft": "",
            "critique": ""
        }
        
        # تشغيل آمن ومضمون باستخدام astream_events أو ainvoke حسب رغبتك
        async for event in graph.astream_events(initial_state, thread, version="v1"):
            pass 
        
        accumulated_text = f"\n\n# مقال عن {task_topic}\n\n"        
        # إظهار الماركداون والزر
        yield "", history, gr.update(visible=True, value=accumulated_text), gr.update(visible=True)
    else:
    # إذا كان سؤالاً عادياً، نضمن إخفاء لوحة المقال والزر تماماً
        yield "", history, gr.update(visible=False, value=""), gr.update(visible=False)
async def continue_graph():
    """إكمال من نقطة التوقف حرفاً بحرف"""
    global accumulated_text
    # جلب العقدة الحالية قبل الإكمال
    state = graph.get_state(thread)
    current_node = state.next[0] if state.next else None
    if not current_node:
        accumulated_text += "\n\n✅ **اكتمل المقال!**"
        yield accumulated_text,gr.update(visible=False)
    
    
    # إضافة عنوان العقدة
    accumulated_text += f"\n\n---\n### {node_names.get(current_node, current_node)}\n"
    
    
    # تنفيذ العقدة الحالية فقط
# تنفيذ العقدة الحالية فقط
    async for event in graph.astream_events(None, thread, version="v1"):
        if event["event"] == "on_chat_model_stream":
                # التعديل البسيط والأضمن هنا:
            chunk_content = event['data']["chunk"].content
            if chunk_content:
                accumulated_text += chunk_content
                yield accumulated_text, gr.update(visible=True)
with gr.Blocks() as demo:
    gr.Markdown("#مساعد ذكي🤖 ")
    chatbot=gr.Chatbot(height=500)

    msg=gr.Textbox(placeholder="اكتب رسالتك هنا من فضلك")

    sendbutton=gr.Button("ارسال",variant="primary")

    addbutton=gr.Button("محادثة جديدة",variant="primary")
    out=gr.Markdown(height=200,visible=False)
    continue_btn = gr.Button("▶️ أكمل", variant="secondary",visible=False)
    msg.submit(chat_gui,[msg,chatbot],[msg,chatbot,out,continue_btn])
    sendbutton.click(chat_gui,[msg,chatbot],[msg,chatbot,out,continue_btn])
    addbutton.click(new_chat,None,[msg,chatbot,out,continue_btn])

    continue_btn.click(
    continue_graph,
    inputs=[],
    outputs=[out,continue_btn]
    )

demo.launch(share=True)


