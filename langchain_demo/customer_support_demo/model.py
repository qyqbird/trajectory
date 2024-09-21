import logging
from typing import Any
from langchain_core.language_models import LLM
from openai import OpenAI


class Kimi(LLM):
    #llm 属性可不定义
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "kimillm"
        
    #必须定义_call方法
    def _call(self, prompt: str, **kwargs: Any) -> str:
    
        try:
          
            client = OpenAI(
                api_key="sk-sChEjX6gKxx7EBx37yLbtu91hQYJDmlJ4IiYB71LvhQzAPAp",
                base_url="https://api.moonshot.cn/v1",
            )
            completion = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                
                temperature=0,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in Kimi _call: {e}", exc_info=True)
            raise


if __name__ == '__main__':
    llm = Kimi()
    res = llm("失业了，好难受")
    print(res)