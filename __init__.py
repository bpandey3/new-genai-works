# Minimal stand-ins for the symbols your code imports
class AwsToAddSOAuth2ClientCredentialsGrantTokenProvider:
    def __init__(self, *args, **kwargs):
        pass
    def get_c2ctoken(self, *args, **kwargs):
        return "fake-token"

class AwsIdpSdkConfig:
    def __init__(self, *args, **kwargs):
        pass


    c2csk/                 ← fake package
      __init__.py
