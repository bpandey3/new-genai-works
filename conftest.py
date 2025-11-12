# tests/conftest.py
import sys
import types

fake = types.ModuleType("c2csk")

class AwsToAddSOAuth2ClientCredentialsGrantTokenProvider:
    def __init__(self, *args, **kwargs):
        pass
    # add whatever methods your code calls in tests:
    def get_c2ctoken(self, *_, **__):
        return "fake-token"

class AwsIdpSdkConfig:
    def __init__(self, *args, **kwargs):
        pass

# expose names at module level so "from c2csk import ..." works
fake.AwsToAddSOAuth2ClientCredentialsGrantTokenProvider = (
    AwsToAddSOAuth2ClientCredentialsGrantTokenProvider
)
fake.AwsIdpSdkConfig = AwsIdpSdkConfig

# register stubbed module before any test import happens
sys.modules["c2csk"] = fake
