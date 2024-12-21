# src/scriptwriting_flow/crews/visual_crew/visual_crew.py
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI

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
        # No context here; we receive inputs from the main flow.
        return Task(
            config=self.tasks_config["generate_visuals"], 
            output_pydantic=VisualPromptSeries
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[self.generate_visuals()],
            process=Process.sequential,
            verbose=True
        )