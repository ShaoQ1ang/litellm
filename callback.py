"""
LiteLLM Proxy 自定义回调处理器

功能：
1. 请求前检查用户余额（precheck），失败会阻塞请求
2. 请求成功后发送费用数据到 callback 服务（失败不阻塞）
3. 请求失败后发送错误数据到 callback 服务（失败不阻塞）

使用方法：
    export LITELLM_CALLBACK_URL="http://localhost:12345"
    export LITELLM_PRECHECK_URL="http://localhost:12345"
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel
import httpx
from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger
import litellm
from litellm.proxy.proxy_server import UserAPIKeyAuth, DualCache


# 配置
CALLBACK_SERVER_URL = os.getenv("LITELLM_CALLBACK_URL", "http://192.168.1.67:12345")
PRECHECK_SERVER_URL = os.getenv("LITELLM_PRECHECK_URL", "http://192.168.1.67:12345")
REQUEST_TIMEOUT = 5.0


# ==================== 数据模型 ====================

class TokenUsage(BaseModel):
    """Token 使用情况"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CallbackData(BaseModel):
    """回调数据模型"""
    request_id: str
    model: str
    messages: List[Dict[str, Any]]
    user: Optional[str] = None
    usage: TokenUsage
    cost: float
    response: Dict[str, Any]
    metadata: Dict[str, Any]
    start_time: str
    end_time: str
    status: str
    exception: str = ""


class LiteLLMCallbackHandler(CustomLogger):
    """LiteLLM 回调处理器"""

    def __init__(self):
        super().__init__()

    # ==================== 辅助方法 ====================

    def _serialize_response(self, response_obj) -> dict:
        """将 ModelResponse 对象转换为可序列化的字典"""
        if response_obj is None:
            return {}
        if isinstance(response_obj, dict):
            return response_obj

        # 尝试 Pydantic v2 的 model_dump
        if hasattr(response_obj, "model_dump"):
            try:
                return response_obj.model_dump(exclude_none=True)
            except Exception:
                pass

        # 尝试 Pydantic v1 的 dict
        if hasattr(response_obj, "dict"):
            try:
                return response_obj.dict(exclude_none=True)
            except Exception:
                pass

        # 降级为字符串
        return {"raw_response": str(response_obj)}

    def _extract_user_info(self, kwargs: dict) -> Dict[str, Any]:
        """从 kwargs 中提取用户信息"""
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {})
        return {
            "user": metadata.get("user_api_key_user_id"),
            "api_key": metadata.get("user_api_key"),
            "request_id": kwargs.get("litellm_call_id"),
        }

    def _build_callback_data(
        self,
        request_id: str,
        model: str,
        messages: list,
        user: str,
        usage: dict,
        cost: float,
        response: dict,
        metadata: dict,
        start_time: datetime,
        end_time: datetime,
        status: str,
        exception: str = "",
    ) -> CallbackData:
        """构建回调数据"""
        return CallbackData(
            request_id=request_id,
            model=model,
            messages=messages,
            user=user,
            usage=TokenUsage(
                # 兼容两种 token 格式：
                # - 对话类型：prompt_tokens, completion_tokens
                # - 图像类型：input_tokens, output_tokens
                prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0),
                completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            cost=cost,
            response=response,
            start_time=start_time.astimezone().isoformat() if start_time else datetime.now().astimezone().isoformat(),
            end_time=end_time.astimezone().isoformat() if end_time else datetime.now().astimezone().isoformat(),
            status=status,
            exception=exception,
        )

    async def _send_callback(self, callback_data: CallbackData) -> bool:
        """
        发送回调数据到 Go 服务器

        Args:
            callback_data: 回调数据模型

        Returns:
            bool: 是否发送成功
        """
        request_id = callback_data.request_id
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{CALLBACK_SERVER_URL}/callback",
                    json=callback_data.model_dump(),
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code == 200:
                    print(f"[Callback] ✓ Success: {request_id}")
                    return True
                else:
                    print(f"[Callback] ✗ Failed: {response.status_code} - {response.text}")
                    return False
        except httpx.ConnectError:
            print(f"[Callback] ✗ Connection Error: Failed to connect to {CALLBACK_SERVER_URL}")
        except httpx.TimeoutException:
            print(f"[Callback] ✗ Timeout: Request timed out")
        except Exception as e:
            print(f"[Callback] ✗ Error: {type(e).__name__}: {e}")
        return False

    # ==================== Proxy Hooks ====================

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ):
        """
        请求前 Hook - 检查余额

        失败时会抛出 HTTPException 阻塞请求
        """
        api_key = getattr(user_api_key_dict, "api_key", None)
        user_id = getattr(user_api_key_dict, "user_id", None)
        request_id = data.get("litellm_call_id")
        model = data.get("model", "")
        messages = data.get("messages", [])

        if not api_key:
            return

        print(f"[PreCheck] Checking: {request_id} | User: {user_id}")

        try:
            # 构建请求体
            request_body = {
                "api_key": api_key,
                "user_id": user_id or "",
                "litellm_request_id": request_id or "",
                "model": model,
            }

            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{PRECHECK_SERVER_URL}/precheck",
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    print(f"[PreCheck] ✓ Passed: {request_id}")
                    return

                # 余额检查失败
                error_data = response.json()
                error_msg = error_data.get("error", "Balance check failed")
                print(f"[PreCheck] ✗ Blocked: {error_msg}")
                raise HTTPException(status_code=402, detail={"error": error_msg})

        except httpx.ConnectError:
            print(f"[PreCheck] ✗ Connection Error: {PRECHECK_SERVER_URL}")
            raise HTTPException(status_code=503, detail={"error": "Service unavailable"})

        except httpx.TimeoutException:
            print(f"[PreCheck] ✗ Timeout: {request_id}")
            raise HTTPException(status_code=504, detail={"error": "Balance check timeout"})

        except HTTPException:
            # 重新抛出 HTTPException
            raise

        except Exception as e:
            print(f"[PreCheck] ✗ Error: {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail={"error": "Internal error"})

    # ==================== CustomLogger 方法 ====================

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        成功事件记录 - 发送费用数据到 Go 服务器

        失败不影响主流程
        """
        try:
            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages", [])

            # 提取用户信息
            user_info = self._extract_user_info(kwargs)
            user = user_info["user"]
            request_id = user_info["request_id"]

            # 获取 metadata
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {})
            # 获取 hidden_params 和 call_type
            hidden_params = getattr(response_obj, '_hidden_params', {})
            call_type = kwargs.get('call_type', 'unknown')

            # 序列化响应
            serialized_response = self._serialize_response(response_obj)
            usage = serialized_response.get("usage", {}) if isinstance(serialized_response, dict) else {}

            # 计算费用 - 根据 call_type 使用不同策略
            is_image_generation = call_type in ('aimage_generation', 'image_generation')
            if is_image_generation:
                # 图像生成：优先使用 hidden_params.response_cost
                response_cost = hidden_params.get('response_cost')
                if response_cost is not None:
                    cost = response_cost
                else:
                    # Fallback: 尝试重新计算
                    try:
                        cost = litellm.completion_cost(completion_response=response_obj)
                    except Exception:
                        cost = 0.0
            else:
                # 其他类型（对话等）
                try:
                    cost = litellm.completion_cost(completion_response=response_obj)
                except Exception:
                    cost = 0.0

            print(f"[Success] {request_id} | Model: {model} | Cost: {cost} | Tokens: {usage.get('total_tokens', 0)}")

            # 构建并发送回调
            callback_data = self._build_callback_data(
                request_id=request_id,
                model=model,
                messages=messages,
                user=user,
                usage=usage,
                cost=cost,
                response=serialized_response,
                metadata=metadata,
                start_time=start_time,
                end_time=end_time,
                status="success",
            )
            await self._send_callback(callback_data)

        except Exception as e:
            print(f"[Success] ✗ Callback Error: {type(e).__name__}: {e}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        失败事件记录 - 发送错误数据到 Go 服务器

        失败不影响主流程
        """
        try:
            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages", [])

            # 提取用户信息
            user_info = self._extract_user_info(kwargs)
            user = user_info["user"]
            request_id = user_info["request_id"]

            # 获取 metadata 和异常信息
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {})
            exception_event = kwargs.get("exception")

            # 提取详细的错误信息
            exception_str = ""
            error_info = metadata.get("error_information", {})

            if error_info:
                # 优先使用 error_message
                error_msg = error_info.get("error_message", "")
                if error_msg:
                    exception_str = error_msg
                else:
                    # 降级使用 error_class
                    error_class = error_info.get("error_class", "")
                    if error_class:
                        exception_str = error_class
                    else:
                        # 最后使用 exception_event
                        exception_str = str(exception_event) if exception_event else ""
            else:
                exception_str = str(exception_event) if exception_event else ""

            # 计算费用
            cost = 0.0
            try:
                if response_obj:
                    cost = litellm.completion_cost(completion_response=response_obj)
            except Exception:
                pass
            # 获取使用情况
            usage = {}
            if response_obj:
                serialized = self._serialize_response(response_obj)
                if isinstance(serialized, dict):
                    usage = serialized.get("usage", {})

            print(f"[Failure] {request_id} | Model: {model} | Exception: {exception_str}")

            # 构建并发送回调
            callback_data = self._build_callback_data(
                request_id=request_id,
                model=model,
                messages=messages,
                user=user,
                usage=usage,
                cost=cost,
                response=self._serialize_response(response_obj),
                metadata=metadata,
                start_time=start_time,
                end_time=end_time,
                status="failure",
                exception=exception_str,
            )
            await self._send_callback(callback_data)

        except Exception as e:
            print(f"[Failure] ✗ Callback Error: {type(e).__name__}: {e}")


# ==================== 创建实例 ====================

def create_handler() -> LiteLLMCallbackHandler:
    """创建回调处理器实例"""
    return LiteLLMCallbackHandler()


proxy_handler_instance = create_handler()
