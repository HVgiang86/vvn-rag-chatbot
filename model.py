class ChatHistory:
    def __init__(self, is_user, content):
        self.is_user = is_user
        self.content = content

    def to_dict(self):
        return {
            'is_user': self.is_user,
            'content': self.content
        }