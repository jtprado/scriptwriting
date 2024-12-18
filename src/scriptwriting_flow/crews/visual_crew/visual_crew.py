from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# from langchain_openai import ChatOpenAI

from scriptwriting_flow.types import VisualPromptSeries

@CrewBase
class VisualCrew():
    """Visual Creation Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    @agent
    def visual_artist(self) -> Agent:
        return Agent(
            config=self.agents_config["visual_artist"],
            verbose=True
        )

    @task
    def generate_visuals(self) -> Task:
        return Task(
            config=self.tasks_config["generate_visuals"], 
            output_pydantic=VisualPromptSeries
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Visual Creation Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )