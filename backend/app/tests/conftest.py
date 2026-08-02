import os


# 测试环境跳过 config.py 的 .env 加载（测试依赖默认配置，不受 .env 调参值干扰）
os.environ.setdefault("RESUMEMIND_SKIP_DOTENV", "1")

# 前台/后台权限分离：测试环境提供管理员密码（security 模块导入期要求非空），
# 并默认关闭限流（大测试集不会被 30 次/分配额打挂；限流单测用 monkeypatch 局部开启）。
os.environ.setdefault("ADMIN_PASSWORD", "test-admin-password")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
