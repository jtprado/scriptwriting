# src/scriptwriting_flow/crews/scriptwriting_crew/scriptwriting_crew.py
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI

from scriptwriting_flow.types import HistoryScriptSeries

llm = LLM(
    model="anthropic/claude-3-5-sonnet-20241022",
    temperature=0.8
)

@CrewBase
class ScriptwritingCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def storyteller(self) -> Agent:
        return Agent(
            config=self.agents_config["storyteller"],
            llm=self.llm,
            verbose=True,
        )

    @task
    def generate_script(self) -> Task:
        # No context needed here because we receive inputs directly from the main flow.
        return Task(
            config=self.tasks_config["script_generation"],
            output_pydantic=HistoryScriptSeries
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[self.generate_script()],
            process=Process.sequential,
            verbose=True,
        )