from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from langchain_openai import ChatOpenAI

from scriptwriting_flow.types import HistoryScriptSeries, HistoryResearchSeries

@CrewBase
class ScriptwritingCrew:
    """Scriptwriting Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def storyteller(self) -> Agent:
        return Agent(
            config=self.agents_config["storyteller"],
            verbose=True,
        )

    @task
    def generate_script(self) -> Task:
        return Task(
            config=self.tasks_config["script_generation"],
            output_pydantic=HistoryScriptSeries,
            context=[HistoryResearchSeries],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Scriptwriting Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )