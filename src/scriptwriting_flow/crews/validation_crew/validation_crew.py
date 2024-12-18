from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# from langchain_openai import ChatOpenAI

from scriptwriting_flow.types import ValidationReport

@CrewBase
class ContentValidationCrew():
    """Content Validation Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def content_critic(self) -> Agent:
        return Agent(
            config=self.agents_config["content_critic"],
            verbose=True
        )

    @task
    def validate_content(self) -> Task:
        return Task(
            config=self.tasks_config["content_validation"],
            output_pydantic=ValidationReport
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Content Validation Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )