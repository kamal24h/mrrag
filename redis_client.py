# redis_client.py
import os
import json
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# اتصال پایدار با ConnectionPool
redis_pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def get_redis_client():
    return redis.Redis(connection_pool=redis_pool)

class ConversationCache:
    @staticmethod
    def create_conversation(redis_client: redis.Redis, conv_id: str, app_id: str, tenant_id: str, user_id: str, ttl_seconds: int = 86400):
        """ذخیره متادیتای مکالمه در ریدیس با زمان انقضا (TTL) 24 ساعته"""
        key = f"conv:{conv_id}:meta"
        data = {
            "conversation_id": conv_id,
            "app_id": app_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "status": "active"
        }
        redis_client.set(key, json.dumps(data), ex=ttl_seconds)

    @staticmethod
    def get_conversation_meta(redis_client: redis.Redis, conv_id: str):
        key = f"conv:{conv_id}:meta"
        raw = redis_client.get(key)
        return json.loads(raw) if raw else None

    @staticmethod
    def append_message(redis_client: redis.Redis, conv_id: str, role: str, content: str):
        """افزودن پیام جدید به لیست پیام‌های محاوره در کش"""
        key = f"conv:{conv_id}:messages"
        msg = {"role": role, "content": content}
        redis_client.rpush(key, json.dumps(msg))

    @staticmethod
    def get_messages(redis_client: redis.Redis, conv_id: str):
        key = f"conv:{conv_id}:messages"
        messages = redis_client.lrange(key, 0, -1)
        return [json.loads(m) for m in messages]

    @staticmethod
    def clear_conversation(redis_client: redis.Redis, conv_id: str):
        redis_client.delete(f"conv:{conv_id}:meta")
        redis_client.delete(f"conv:{conv_id}:messages")