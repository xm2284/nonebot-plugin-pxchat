from openai import BadRequestError, AsyncOpenAI
from nonebot import logger
from .manager import chat_manager
from .mcp_manager import mcp_client  # 导入MCP管理器
import asyncio
import json

def get_current_time() -> str:
    """获取当前时间"""
    import datetime
    now = datetime.datetime.now()
    return f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}"

# 定义本地工具列表
local_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {
                "type": "object", 
                "properties": {},
                "required": []
            },
        }
    }
]

# 本地工具调用映射
local_available_functions = {
    "get_current_time": get_current_time
}

async def get_chat_reply_with_tools(messages: list, is_group: bool = False) -> str:
    """
    结合function call和分段回复的聊天回复函数 - 使用消息副本处理工具调用
    """
    # 检查全局开关
    if not chat_manager.is_chat_enabled():
        raise Exception("聊天功能当前已关闭")
    
    # 检查MCP功能是否启用
    if not chat_manager.is_mcp_enabled():
        logger.info("MCP功能未启用，使用普通聊天模式")
        return await get_chat_reply(messages, is_group)
    
    # 获取当前AI配置
    ai_config = chat_manager.get_current_ai_config()
    
    if not ai_config:
        raise Exception("未配置服务，请使用 'px ai add' 命令添加配置")
    
    try:
        # 创建消息副本用于工具调用处理
        processing_messages = messages.copy()
        
        # 获取工具列表
        all_tools = local_tools.copy()  # 先添加本地工具
        
        # 检查是否有启用的MCP服务器
        enabled_servers = chat_manager.get_enabled_mcp_servers()
        if enabled_servers:
            try:
                mcp_tools_list = await mcp_client.get_tools()
                mcp_tools = mcp_client.get_openai_tools_format()
                all_tools.extend(mcp_tools)
                logger.info(f"MCP功能已启用，可用工具总数: {len(all_tools)} (本地: {len(local_tools)}, MCP: {len(mcp_tools)})")
            except Exception as e:
                logger.warning(f"获取MCP工具失败，将只使用本地工具: {e}")
        else:
            logger.info("没有启用的MCP服务器，只使用本地工具")
        
        # 动态创建客户端
        client = AsyncOpenAI(
            api_key=ai_config.get("api_key", ""),
            base_url=ai_config.get("api_url", ""),
        )
        
        # 第一阶段：Function Call处理（使用异步调用）
        logger.info("开始工具调用判断")
        response = await client.chat.completions.create(
            model=ai_config.get("model", ""),
            messages=[
                {
                    "role": "user",
                    "content": "请仅判断是否需要调用工具，若需要则直接调用，不需要则回复NO\n" +
                                f"问题: {processing_messages[-1]['content']}"
                }
            ],
            tools=all_tools,
            tool_choice="auto",
            max_tokens=128,
        )
        
        message = response.choices[0].message
        tool_calls = message.tool_calls
        
        # 记录Token消耗
        if hasattr(response, 'usage') and response.usage:
            usage_info = response.usage
            prompt_tokens = getattr(usage_info, 'prompt_tokens', 0)
            completion_tokens = getattr(usage_info, 'completion_tokens', 0)
            total_tokens = getattr(usage_info, 'total_tokens', 0)
            logger.info(f"Function Call判断Token消耗 - 提示Token: {prompt_tokens}, 补全Token: {completion_tokens}, 总计: {total_tokens}")
        
        # 如果有工具调用，执行调用（在副本上进行）
        if tool_calls:
            logger.info("检测到工具调用，开始执行函数")
            
            # 将模型的回复添加到消息副本中
            processing_messages.append({
                "role": "assistant",
                "content": message.content if message.content else "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    } for tool_call in tool_calls
                ]
            })
            
            # 处理每个工具调用
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logger.info(f"调用函数: {function_name}, 参数: {function_args}")
                
                # 判断是本地工具还是MCP工具
                if function_name in local_available_functions:
                    # 调用本地函数
                    function_result = local_available_functions[function_name](**function_args)
                    logger.info(f"本地函数结果: {function_result}")
                    
                    processing_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(function_result)
                    })
                else:
                    # 调用MCP工具
                    try:
                        mcp_result = await mcp_client.call_tool(function_name, function_args)
                        processing_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": str(mcp_result)
                        })
                    except Exception as e:
                        error_msg = f"MCP工具调用失败: {str(e)}"
                        processing_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id, 
                            "name": function_name,
                            "content": error_msg
                        })
        
        # 第二阶段：使用处理后的消息副本生成最终回复
        tmp_record = "\n".join(str(msg) for msg in processing_messages[-5:])
        logger.info(f"最近5条处理消息记录（含工具调用过程）:\n{tmp_record}")
        # 使用 processing_messages（包含工具调用过程）来生成最终回复
        logger.info(f"调用原有逻辑生成格式化回复")
        final_reply = await get_chat_reply(processing_messages, is_group)
        
        # 但只将最终回复添加到原始消息中，不包含工具调用过程
        return final_reply
        
    except Exception as e:
        logger.error(f"get_chat_reply_with_tools 发生异常: {e}")
        # 如果工具调用失败，回退到普通聊天模式
        logger.info("工具调用失败，回退到普通聊天模式")
        return await get_chat_reply(messages, is_group)

def get_reply_format(is_group: bool = False):
    base_format = ""
    if is_group:
        base_format += """
你现在在群聊中，请注意：
    - 你的发言会出现在群聊中，所有成员都能看到
    - 保持自然，像普通群成员一样参与讨论
    - 不要表现得像机器人，要有自己的观点和性格
"""
    base_format += """
请严格按照以下JSON格式回复，不要包含任何其他内容：
{
  "reply": [
    "第一段内容",
    "第二段内容",
    "可能还有更多段..."
  ]
}

内容要求：
1. 不要使用markdown格式，只返回纯JSON
2. 回复段数随机，80%的情况下保持一段内容，保持简洁
3. 在以下情况下必须分段：
   - 内容明显切换主题（比如从问题分析转到个人建议）
   - 包含代码块、示例或需要突出显示的部分
   - 回复较长时，分段模仿自然停顿，像网友打字时的换行习惯
4. 每个段落应该是一个完整的句子或者语义单元，结尾不要出现句号
5. 如果是一段代码，保持代码完整作为一个段落
6. 整体风格贴近真实网友：多用'我'开头，带点小错误或口语化表达（如'可能吧'、'反正我觉得'），但别过度啰嗦
"""    
    return base_format

def get_system_prompt(is_group: bool = False):
    personality = chat_manager.get_personality()
    return personality + get_reply_format(is_group)

async def get_chat_reply(messages: list, is_group: bool = False) -> str:
    """
    messages: [{"role": "user|assistant|system", "content": str}, ...]
    is_group: 是否为群聊环境
    """
    # 检查全局开关
    if not chat_manager.is_chat_enabled():
        raise Exception("聊天功能当前已关闭")
    
    # 获取当前AI配置
    ai_config = chat_manager.get_current_ai_config()
    
    if not ai_config:
        raise Exception("未配置服务，请使用 'px ai add' 命令添加配置")
    
    try:
        # 动态创建客户端
        client = AsyncOpenAI(
            api_key=ai_config.get("api_key", ""),
            base_url=ai_config.get("api_url", ""),
        )
        
        # 构建请求参数
        request_params = {
            "model": ai_config.get("model", ""),
            "messages": [{"role": "system", "content": get_system_prompt(is_group)}] + messages,
            "response_format": {
                'type': 'json_object'
            }
        }
        
        # 只有在搜索功能启用时才添加搜索参数
        if chat_manager.is_search_enabled():
            request_params["extra_body"] = {
                "enable_search": True,
                "search_options": {"forced_search": True}
            }
        
        # 直接使用异步调用
        reply_obj = await client.chat.completions.create(**request_params)
        reply = reply_obj.choices[0].message.content
        
        # 获取并记录Token消耗
        if hasattr(reply_obj, 'usage') and reply_obj.usage:
            usage_info = reply_obj.usage
            prompt_tokens = getattr(usage_info, 'prompt_tokens', 0)
            completion_tokens = getattr(usage_info, 'completion_tokens', 0)
            total_tokens = getattr(usage_info, 'total_tokens', 0)
            logger.info(f"对话Token消耗 - 提示Token: {prompt_tokens}, 补全Token: {completion_tokens}, 总计: {total_tokens}")

        if not reply:
            raise Exception("AI返回了空回复")
            
        return reply
        
    except BadRequestError as e:
        # 处理请求错误
        error_msg = f"对话请求异常\n{e}"
        raise Exception(error_msg)
    except Exception as e:
        # 重新抛出其他异常
        raise e

async def should_reply_in_group(messages: list) -> bool:
    """
    判断在群聊中是否应该回复（当没有被@时）
    """
    # 获取当前AI配置
    ai_config = chat_manager.get_current_ai_config()
    
    if not ai_config:
        return False
    
    try:
        # 构建判断提示词
        judgment_prompt = """
你是一个在群聊中的参与者，需要判断是否要主动参与对话。请基于以下原则判断：

【需要回复的情况】
1. 有人直接发出提问或寻求建议（即使没at你）
2. 有人表达了困惑或需要帮助
3. 有人分享有趣内容，适合互动回应
4. 话题与你相关或你有独特见解

【不需要回复的情况】
1. 其他人正在相互对话
2. 话题与你完全无关
4. 对话已经有很多人参与，不缺互动
5. 如果出现了at的内容，注意不是at你

请分析最近的对话，判断是否需要你参与
只回复 "YES" 或 "NO"，不要其他内容。
"""
        
        client = AsyncOpenAI(
            api_key=ai_config.get("api_key", ""),
            base_url=ai_config.get("api_url", ""),
        )
        
        judge_content = []
        for msg in messages[-10:]:
            if msg["role"] == "user":
                judge_content.append(f"{msg['content']}")
            else:
                try:
                    data = json.loads(msg['content'])
                    reply = data.get('reply', ['']) if isinstance(data, dict) else msg['content']
                except (json.JSONDecodeError, TypeError):
                    reply = msg['content']
                judge_content.append(f"你(px)回复说: {reply}")
        content = "\n".join(judge_content)
        
        completion_obj = await client.chat.completions.create(
            model=ai_config.get("model", ""),
            messages=[{"role": "system", "content": chat_manager.get_personality() + judgment_prompt}, {"role": "user", "content": "群聊记录\n" + content}],
            max_tokens=10
        )
        
        judgment = completion_obj.choices[0].message.content

        # 获取并记录Token消耗
        if hasattr(completion_obj, 'usage') and completion_obj.usage:
            usage_info = completion_obj.usage
            prompt_tokens = getattr(usage_info, 'prompt_tokens', 0)
            completion_tokens = getattr(usage_info, 'completion_tokens', 0)
            total_tokens = getattr(usage_info, 'total_tokens', 0)
            logger.info(f"判断Token消耗 - 提示Token: {prompt_tokens}, 补全Token: {completion_tokens}, 总计: {total_tokens}")

        logger.info(f"群聊回复判断结果: {judgment.strip().upper()}")

        judgment = judgment.strip().upper() if judgment else "NO"
        
        return judgment == "YES"
        
    except Exception as e:
        raise e