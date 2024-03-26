"""Hypothetical Document Embeddings.

https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Extra

from langchain.chains.base import Chain
from langchain.chains.hyde.prompts import PROMPT_MAP
from langchain.chains.llm import LLMChain

from retry import retry
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document


class HypotheticalDocumentEmbedder(Chain, Embeddings):
    """Generate hypothetical document for query, and then embed that.

    Based on https://arxiv.org/abs/2212.10496
    """

    base_embeddings: Embeddings
    llm_chain: ChatOpenAI

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Input keys for Hyde's LLM chain."""
        # replace with placeholder for abstract method
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys for Hyde's LLM chain."""
        # replace with placeholder for abstract method
        return ["output"]


    def embed_documents(self, texts: List[str], hypo_params) -> List[List[float]]:
        """Call the base embeddings."""
        if hypo_params['verbose']:
            print("\n>>> Generating embeddings for hypothetical documents...\n")
        return self.base_embeddings.embed_documents(texts)

    def combine_embeddings(self, embeddings: List[List[float]], hypo_params) -> List[float]:
        """Combine embeddings into final embeddings."""
        if hypo_params['verbose']:
            print("\n>>> Combining embeddings for hypothetical documents...\n")
        return list(np.array(embeddings).mean(axis=0))

    # ORIGINAL IMPLEMENTATION
#     def embed_query(self, text: str) -> List[float]:
#         """Generate a hypothetical document and embedded it."""
#         var_name = self.llm_chain.input_keys[0]
#         result = self.llm_chain.generate([{var_name: text}])
#         documents = [generation.text for generation in result.generations[0]]
#         for ii, doc in enumerate(documents):
#             print(f"### Hyde Document {ii+1} ###\n{doc}\n")
#         embeddings = self.embed_documents(documents)
#         return self.combine_embeddings(embeddings)
    
    # AUTO HYDE IMPLEMENTATION
    def embed_query(
        self, 
        text: str, 
        db: VectorStore, 
        hypo_params: dict
        ) -> List[float]:

        keywords = self.extract_keywords(text, hypo_params)
        init_docs = self.do_init_retrieval(db, text, hypo_params)
        remaining_docs = self.get_remaining_docs_with_keywords(text, init_docs, keywords, hypo_params)
        cat_dict = self.cluster_docs(remaining_docs, hypo_params)
        hypo_docs = self.generate_hypo_docs(text, cat_dict, hypo_params)
        embeddings = self.embed_documents(hypo_docs, hypo_params)
        combined_embeddings = self.combine_embeddings(embeddings, hypo_params)

        if hypo_params['verbose']:
            print("\n>>> Auto Hyde Embedding Complete!\n")
        return combined_embeddings
    
    def do_init_retrieval(
        self,
        db: VectorStore, 
        text: str, 
        hypo_params: dict
        ) -> List[Tuple[Document, float]]:
        
        k = hypo_params['baseline_k'] * hypo_params['exploration_multiplier']
        if hypo_params['verbose']:
            print(f"\n>>> Performing Initial Retrieval of {k} documents...\n")
        docs = db.similarity_search_with_score(
            text, 
            k=k
        )
        return docs
    
    @retry(tries=5)
    def extract_keywords(
        self, 
        text: str, 
        hypo_params: dict
        ) -> List[str]:

        if hypo_params['verbose']:
            print(f"\n>>> Extracting Keywords from your Query...")
        
        KEYWORD_EXTRACTION_PROMPT = """
        Your goal is to extract a list of keywords from an input phrase, sentence, or several sentences.

        - You can only generate 1 to 5 keywords.
        - Keywords should be nouns, issues, concepts
        - Keywords should not include verbs, prepositions, pronouns
        - Each keyword can only be one word long.
        - If the input is just a single word, return that word as the only keyword.

        {format_instructions}

        The input is:
        {input}
        """

        class KeywordListSchema(BaseModel):
            keywordList: list[str] = Field(description="list of one-word keywords based on a given phrase")

        parser = JsonOutputParser(pydantic_object=KeywordListSchema)

        prompt = ChatPromptTemplate.from_template(
            template=KEYWORD_EXTRACTION_PROMPT,
            intput_variables = ["input"],
            partial_variables = {
                'format_instructions': parser.get_format_instructions()
            }
        )

        keyword_extraction_chain = (
            {'input': RunnablePassthrough()}
            | prompt
            | self.llm_chain
            | parser
        )
        
        keywords = keyword_extraction_chain.invoke(text)['keywordList']
        if hypo_params['verbose']:
            print(f">>> ...Keywords Extracted: {keywords}\n")
        
        return keywords

    def get_remaining_docs_with_keywords(
        self, 
        text: str, 
        init_docs: List[Tuple[Document, float]], 
        keywords: List[str], 
        hypo_params: dict
        ) -> List[Document]:

        remaining_docs_with_keywords = list()
        
        if hypo_params['verbose']:
            print(f"\n>>> Checking {len(init_docs[hypo_params['baseline_k']:])} Docs ranked after {hypo_params['baseline_k']} for presence of keyword...")

        for r in init_docs[hypo_params['baseline_k']:]:
            page_content = r[0].page_content.lower()
            for keyword in keywords:
                if keyword.lower() in page_content:
                    remaining_docs_with_keywords.append(r)
                    continue
                    
        if hypo_params['verbose']:
            print(f">>> ...{len(remaining_docs_with_keywords)} neglected Docs identified\n")
        return remaining_docs_with_keywords

    def cluster_docs(
        self, 
        remaining_docs_with_keywords: List[Document], 
        hypo_params: dict
        ) -> Dict[int, List[str]]:
        
        from hdbscan import HDBSCAN
        
        if hypo_params['verbose']:
            print(f"\n>>> Clustering neglected Docs...")
        
        embeddings = self.embed_documents([
            r[0].page_content 
            for r in remaining_docs_with_keywords], 
            {'verbose': False})
        hdb = HDBSCAN(min_samples=1, min_cluster_size=3).fit(embeddings)
        remaining_docs_with_cat = filter(lambda x: x[1] != -1, zip([r[0].page_content for r in remaining_docs_with_keywords], hdb.labels_))
        
        cat_dict = {}

        for page_content, cat in remaining_docs_with_cat:
            if cat not in cat_dict:
                cat_dict[cat] = [page_content]
            else:
                cat_dict[cat].append(page_content)
                
        if hypo_params['verbose']:
            print(f">>> ...{len(cat_dict)} Clusters identified\n")
                
        return cat_dict
    
    @retry(tries=5)
    def generate_hypo_docs(
        self, 
        text: str, 
        cat_dict: Dict[int, List[str]], 
        hypo_params: dict
        ) -> List[str]:
        
        hypo_docs = list()
        
        if hypo_params['verbose']:
            print(f"\n>>> Generating Hypothetical Documents for each Doc Cluster...\n")

        HYPOTHETICAL_DOCUMENT_PROMPT = """
        Your instruction is to generate a single hypothetical document from an input.
        - This hypothetical document must be similar in style, tone and voice as examples you are provided with.
        - This hypothetical document must appear like it was written by the same author as the examples you are provided with.
        - This hypothetical document must also be similar in length with the examples you are provided with.

        {format_instructions}

        ### EXAMPLES ###
        Below are some examples of hypothetical documents, all written by the same author, in pairs of <Input> and <Hypothetical Document>:

        {ref_documents}

        ### INSTRUCTION ###
        Now generate a new hypothetical document. 

        <Input>
        {input}
        <Hypothetical Document>

        """

        class HypotheticalDocumentSchema(BaseModel):
            hypotheticalDocument: str = Field(description="a hypothetical document given an input word, phrase or question")

        parser = JsonOutputParser(pydantic_object=HypotheticalDocumentSchema)

        prompt = ChatPromptTemplate.from_template(
            template=HYPOTHETICAL_DOCUMENT_PROMPT,
            intput_variables = ["input", "ref_documents"],
            partial_variables = {
                'format_instructions': parser.get_format_instructions()
            }
        )

        hypothetical_document_chain = (
            {'input': RunnablePassthrough(), 'ref_documents': RunnablePassthrough()}
            | prompt
            | self.llm_chain
            | parser
        )

        cat_ii = 1
        for cat in cat_dict.keys():

            ref_doc_string = ""
            doc_ii = 1
            for doc in cat_dict[cat]:
                ref_doc_string += f"\n\n<Input>"
                ref_doc_string += text
                ref_doc_string += f"\n\n<Hypothetical Document>\n"
                ref_doc_string += f'{{"hypotheticalDocument": "{doc}"}}'
                doc_ii += 1

            hypo_doc = hypothetical_document_chain.invoke(
                {'input': text, 'ref_documents': ref_doc_string}
            )['hypotheticalDocument']

            if hypo_params['verbose']:
                print(f"\n### Hypo Doc {cat_ii} ###")
                print(hypo_doc+'\n')
            
            hypo_docs.append(hypo_doc)
            
            cat_ii += 1
            
        return hypo_docs

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Call the internal llm chain."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        return self.llm_chain(inputs, callbacks=_run_manager.get_child())

    # NOT REQUIRED
#     @classmethod
#     def from_llm(
#         cls,
#         llm: BaseLanguageModel,
#         base_embeddings: Embeddings,
#         prompt_key: Optional[str] = None,
#         custom_prompt: Optional[BasePromptTemplate] = None,
#         **kwargs: Any,
#     ) -> HypotheticalDocumentEmbedder:
#         """Load and use LLMChain with either a specific prompt key or custom prompt."""
#         if custom_prompt is not None:
#             prompt = custom_prompt
#         elif prompt_key is not None and prompt_key in PROMPT_MAP:
#             prompt = PROMPT_MAP[prompt_key]
#         else:
#             raise ValueError(
#                 f"Must specify prompt_key if custom_prompt not provided. Should be one "
#                 f"of {list(PROMPT_MAP.keys())}."
#             )

#         llm_chain = LLMChain(llm=llm, prompt=prompt)
#         print(f"### Hyde Propmt ###\n{prompt}\n")
#         return cls(base_embeddings=base_embeddings, llm_chain=llm_chain, **kwargs)

    @property
    def _chain_type(self) -> str:
        return "hyde_chain"