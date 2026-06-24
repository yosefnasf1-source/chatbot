from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated, Literal
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, ToolMessage
from langchain_openai import ChatOpenAI
import gradio as gr
import operator
from pydantic import BaseModel, Field

# ============================================================
# 1. البيانات الأساسية
# ============================================================

PRODUCTS = {
    "laptop": 1299.99,
    "headphones": 149.95,
    "keyboard": 89.50,
    "mouse": 49.99,
    "monitor": 399.99,
    "speaker": 79.99
}

DISCOUNTS = {"bronze": 10, "silver": 15, "gold": 20}

# ============================================================
# 2. Pydantic Classes
# ============================================================

class ProductInput(BaseModel):
    product: Literal["laptop", "headphones", "keyboard", "mouse", "monitor", "speaker"] = Field(description="name of product")

class DiscountInput(BaseModel):
    price: float = Field(description="the price of product", gt=0)
    discount_tier: Literal["bronze", "silver", "gold"] = Field(
        description="the type of discount. MUST be bronze, silver, or gold."
    )

class PricesInput(BaseModel):
    prices: List[float] = Field(
        description="A list of product prices that the customer has requested.",
        min_items=1
    )

class DiscountCalculationInput(BaseModel):
    subtotal: float = Field(
        description="Total price before any discount (sum of all original prices of products which 'get_product_price' tool return it)",
        gt=0
    )
    total_after_discount: float = Field(
        description="Total price after applying discounts prices of products which 'get_product_price' tool return it after the discount",
        ge=0
    )

# ============================================================
# 3. الأدوات
# ============================================================

@tool(args_schema=ProductInput)
def get_product_price(product: str) -> float:
    """Get the price of a product."""
    return PRODUCTS[product]

@tool(args_schema=DiscountInput)
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price."""
    discount = DISCOUNTS[discount_tier]
    return round(price - (price * discount / 100), 2)

@tool(args_schema=PricesInput)
def sum_prices(prices: List[float]) -> float:
    """
    Sum a list of prices (numbers) and return the total sum.
    This tool takes a list of numbers only.
    """
    total = sum(prices)
    return round(total, 2)

@tool(args_schema=DiscountCalculationInput)
def calculate_discount_amount(subtotal: float, total_after_discount: float) -> str:
    """
    Calculate the total discount amount and percentage from subtotal and total after discount.
    """
    discount_amount = subtotal - total_after_discount
    discount_percentage = (discount_amount / subtotal * 100)
    
    result = f"""💸 **Discount Amount:** ${discount_amount:.2f}
📉 **Discount Percentage:** {discount_percentage:.1f}%"""
    
    return result

# ============================================================
# 4. الـ Graph
# ============================================================

SYSTEM_PROMPT = """You are a friendly shopping assistant.

Products: laptop, headphones, keyboard, mouse, monitor, speaker.
Discounts: bronze (10%), silver (15%), gold (20%).

Workflow:
1. For each product:
   - Call get_product_price (only once per product type; reuse known price if repeated)
   - IF customer mentions a discount (bronze/silver/gold), call apply_discount on that product
   - Store original and discounted prices in separate lists
2. After all products:
   - Call sum_prices on original list → subtotal
   - Call sum_prices on discounted list → total after discount
   - Call calculate_discount_amount → discount summary

Rules:
- Apply discount PER product, NOT on total
- ONLY call apply_discount if customer mentions bronze, silver, or gold
- If no discount mentioned, ASK: "Do you have a discount card? (bronze, silver, gold)"
- If customer says NO or NONE → DO NOT call apply_discount at all
- **NEVER pass any value to apply_discount unless it's bronze, silver, or gold**
- ALWAYS use calculate_discount_amount
- **CACHE POLICY: If the same tool with the exact same parameters has been called before in this conversation, DO NOT call it again. Reuse the previously returned result instead.**
- **DOUBLE-CHECK: Before calling sum_prices, ensure both lists have the same number of items and correspond to the same products.**

Show: each product price, subtotal, discount amount, final total.

Answer in client language."""
class State(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

model = ChatOpenAI(model="gpt-4o", temperature=0)
tools = {t.name: t for t in [get_product_price, apply_discount, sum_prices, calculate_discount_amount]}
agent_model = model.bind_tools([get_product_price, apply_discount, sum_prices, calculate_discount_amount])

def agent(state: State):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [agent_model.invoke(messages)]}

def is_tools_con(state: State):
    return bool(state['messages'][-1].tool_calls)

def do_action(state: State):
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        if t["name"] in tools:
            res = tools[t["name"]].invoke(t["args"])
            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(res)))
    return {"messages": results}

builder = StateGraph(State)
builder.add_node("llm", agent)
builder.add_node("action", do_action)
builder.set_entry_point("llm")
builder.add_conditional_edges("llm", is_tools_con, {True: "action", False: END})
builder.add_edge("action", "llm")
graph = builder.compile(checkpointer=MemorySaver())

# ============================================================
# 5. واجهة Gradio
# ============================================================

thread_count = 0

def get_products_display():
    return "### 🛍️ المنتجات\n" + "\n".join([f"- {k}: ${v}" for k, v in PRODUCTS.items()])

def get_discounts_display():
    return "### 🎫 الخصومات\n" + "\n".join([f"- {k}: {v}%" for k, v in DISCOUNTS.items()])

def new_chat():
    global thread_count
    thread_count += 1
    return "", [], ""

async def chat_gui(message, history):
    global thread_count
    
    print("\n" + "🟢" * 30)
    print(f"💬 سؤال جديد: {message}")
    print("🟢" * 30 + "\n")
    
    thread = {"configurable": {"thread_id": f"user_{thread_count}"}}
    if not message or not message.strip():
        yield "", history, "⚠️ اكتب رسالة"
        return
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    full = ""
    
    async for event in graph.astream_events({"messages": [HumanMessage(content=message)]}, thread, version="v2"):
        
        # التقاط انتهاء تنفيذ أداة
        if event["event"] == "on_tool_end":
            tool_name = event["name"]
            tool_input = event["data"].get("input", {})
            tool_output = event["data"].get("output", "")
            
            print("\n" + "─" * 50)
            print(f"🔧 الأداة: {tool_name}")
            print(f"   📥 البارامترات: {tool_input}")
            print(f"   📤 النتيجة: {tool_output}")
            print("─" * 50)
        
        # التقاط تدفق النموذج (الرد)
        elif event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                full += chunk
                history[-1]["content"] = full
                yield "", history, ""

    yield "", history, "✅ تم"

with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# 🛒 نظام التسوق الذكي")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(get_products_display())
            gr.Markdown("---")
            gr.Markdown(get_discounts_display())
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(placeholder="مثال: أريد لابتوب مع خصم فضية")
            send_btn = gr.Button("إرسال", variant="primary")
            new_btn = gr.Button("🔄 جديد", variant="secondary")
            status = gr.Markdown("")
    msg.submit(chat_gui, [msg, chatbot], [msg, chatbot, status])
    send_btn.click(chat_gui, [msg, chatbot], [msg, chatbot, status])
    new_btn.click(new_chat, None, [msg, chatbot, status])

demo.launch(share=True)