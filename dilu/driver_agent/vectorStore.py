import os
import textwrap
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document

from dilu.scenario.envScenario import EnvScenario


class DrivingMemory:

    def __init__(self, encode_type='sce_language', db_path=None) -> None:
        self.encode_type = encode_type
        if encode_type == 'sce_encode':
            # 'sce_encode' is deprecated for now.
            raise ValueError("encode_type sce_encode is deprecated for now.")
        elif encode_type == 'sce_language':
            if os.environ["OPENAI_API_TYPE"] == 'azure':
                self.embedding = OpenAIEmbeddings(
                    deployment=os.environ['AZURE_EMBED_DEPLOY_NAME'], chunk_size=1)
            elif os.environ["OPENAI_API_TYPE"] == 'openai':
                self.embedding = OpenAIEmbeddings()
            else:
                raise ValueError(
                    "Unknown OPENAI_API_TYPE: should be azure or openai")
            db_path = os.path.join(
                './db', 'chroma_5_shot_20_mem/') if db_path is None else db_path
            self.scenario_memory = Chroma(
                embedding_function=self.embedding,
                persist_directory=db_path
            )
        else:
            raise ValueError(
                "Unknown ENCODE_TYPE: should be sce_encode or sce_language")
        print("==========Loaded ",db_path," Memory, Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.==========")

    def retriveMemory(self, driving_scenario: EnvScenario, frame_id: int, top_k: int = 5):
        if self.encode_type == 'sce_encode':
            pass
        elif self.encode_type == 'sce_language':
            query_scenario = driving_scenario.describe(frame_id)
            similarity_results = self.scenario_memory.similarity_search_with_score(
                query_scenario, k=top_k)
            fewshot_results = []
            for idx in range(0, len(similarity_results)):
                # print(f"similarity score: {similarity_results[idx][1]}")
                fewshot_results.append(similarity_results[idx][0].metadata)
        return fewshot_results

    def addMemory(self, sce_descrip: str, human_question: str, response: str, action: int, sce: EnvScenario = None, comments: str = ""):
        if self.encode_type == 'sce_encode':
            pass
        elif self.encode_type == 'sce_language':
            sce_descrip = sce_descrip.replace("'", '')
        # https://docs.trychroma.com/usage-guide#using-where-filters
        get_results = self.scenario_memory._collection.get(
            where_document={
                "$contains": sce_descrip
            }
        )
        # print("get_results: ", get_results)

        if len(get_results['ids']) > 0:
            # already have one
            id = get_results['ids'][0]
            self.scenario_memory._collection.update(
                ids=id, metadatas={"human_question": human_question,
                                   'LLM_response': response, 'action': action, 'comments': comments}
            )
            print("Modify a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
        else:
            doc = Document(
                page_content=sce_descrip,
                metadata={"human_question": human_question,
                          'LLM_response': response, 'action': action, 'comments': comments}
            )
            id = self.scenario_memory.add_documents([doc])
            print("Add a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    def deleteMemory(self, ids):
        self.scenario_memory._collection.delete(ids=ids)
        print("Delete", len(ids), "memory items. Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    def combineMemory(self, other_memory):
        other_documents = other_memory.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        current_documents = self.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        for i in range(0, len(other_documents['embeddings'])):
            if other_documents['embeddings'][i] in current_documents['embeddings']:
                print("Already have one memory item, skip.")
            else:
                self.scenario_memory._collection.add(
                    embeddings=other_documents['embeddings'][i],
                    metadatas=other_documents['metadatas'][i],
                    documents=other_documents['documents'][i],
                    ids=other_documents['ids'][i]
                )
        print("Merge complete. Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")


if __name__ == "__main__":
    pass