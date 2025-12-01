from fastapi import FastAPI
from pydantic import BaseModel

from agent import run_agent


app = FastAPI()


class PipelineFailure(BaseModel):
    pod_name: str
    container_name: str
    namespace: str


@app.get('/health')
async def health():
    print('receive health request')
    return 'ok'


@app.post("/report-failure")
async def report_failure(pipeline_failure: PipelineFailure):
    print(f'received report-failure request with payload: '
          f'{pipeline_failure}')
    try:
        run_agent(
            pipeline_failure.pod_name, pipeline_failure.namespace
        )
    except Exception as e:
        print(f'encountered error: {e}')
    print('responding to client')
    return pipeline_failure