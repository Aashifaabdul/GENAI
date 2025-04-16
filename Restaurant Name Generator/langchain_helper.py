from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain #output of one is an input for others so use chain 
from langchain.chains import SequentialChain

llm = Ollama(model="mistral", temperature=0.7)


def generate_restaurant_name(cuisine):
    prompt_template_name=PromptTemplate(
    input_variables=['cuisine'],
    template='I want to opean a restaurant for {cuisine} food . suggest ONE fancy name for my restaurant.only ONE name please,give me the name alone ')
    name_chain =LLMChain(llm=llm,prompt=prompt_template_name,output_key="restaurant_name")

    prompt_template_ITEMS=PromptTemplate(
    input_variables=['restaurant_name'],
    template='suggest some menu items for {restaurant_name}. Return only their names (i dont want any descriptions) as a comma seperated values ')
    food_items_chain =LLMChain(llm=llm,prompt=prompt_template_ITEMS,output_key="menu_items") 
    chain=SequentialChain(chains=[name_chain,food_items_chain],
                      input_variables=['cuisine'],
                      output_variables=["restaurant_name","menu_items"])
    response=chain({'cuisine':cuisine})
    
    return response

