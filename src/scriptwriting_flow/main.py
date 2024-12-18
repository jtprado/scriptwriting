import os
from typing import List
from pydantic import BaseModel

from crewai.flow.flow import Flow, listen, start

from scriptwriting_flow.crews.research_crew.research_crew import ResearchCrew
from scriptwriting_flow.crews.scriptwriting_crew.scriptwriting_crew import ScriptwritingCrew
from scriptwriting_flow.crews.visual_crew.visual_crew import VisualCrew
from scriptwriting_flow.crews.validation_crew.validation_crew import ContentValidationCrew
from scriptwriting_flow.types import (
    HistoryResearchSeries, 
    HistoryScriptSeries, 
    VisualPromptSeries, 
    ValidationReport
)


class ScriptFlowState(BaseModel):
    topic: str = "Invention of Radio"
    age_range: str = "18-30"
    target_audience: str = "male history enthusiasts who enjoy history told like it happened"
    interests: List[str] = ["history", "science", "technology"]
    research_series: HistoryResearchSeries | None = None
    script_series: HistoryScriptSeries | None = None
    visuals_series: VisualPromptSeries | None = None
    validation_report: ValidationReport | None = None
    iteration_done: bool = False  # Track if we already did a revision pass


class ScriptFlow(Flow[ScriptFlowState]):
    """Flow to write multi-part scripts based on a given topic."""

    initial_state = ScriptFlowState

    @start()
    def perform_research(self):
        print("Starting Research Crew")
        research_crew = ResearchCrew().crew()
        # Now we directly pass user input to deep research, then split into 3 parts
        research_output = research_crew.kickoff(
            inputs={
                "topic": self.state.topic,
                "age_range": self.state.age_range,
                "target_audience": self.state.target_audience,
                "interests": self.state.interests,
            }
        )
        # The final output of the research crew is a HistoryResearchSeries
        self.state.research_series = research_output
        return research_output

    @listen("perform_research")
    def write_script(self):
        print("Starting Scriptwriting Crew")
        inputs = {
            "research_parts": [part.dict() for part in self.state.research_series.parts],
            "target_audience": self.state.target_audience,
        }

        if self.state.validation_report and self.state.validation_report.improvement_suggestions:
            inputs["critic_feedback"] = self.state.validation_report.improvement_suggestions

        scriptwriting_crew = ScriptwritingCrew().crew()
        script_output = scriptwriting_crew.kickoff(inputs=inputs)
        self.state.script_series = script_output
        return script_output

    @listen("write_script")
    def create_visuals(self):
        print("Starting Visual Creation Crew")
        visual_crew = VisualCrew().crew()
        visuals_output = visual_crew.kickoff(
            inputs={
                "scripts_parts": [script.dict() for script in self.state.script_series.parts],
                "target_audience": self.state.target_audience,
            }
        )
        self.state.visuals_series = visuals_output
        return visuals_output

    @listen("create_visuals")
    def validate_content(self):
        print("Starting Content Validation Crew")
        validation_crew = ContentValidationCrew().crew()
        validation_output = validation_crew.kickoff(
            inputs={
                "scripts_output": [script.dict() for script in self.state.script_series.parts],
                "visuals_output": [v.dict() for v in self.state.visuals_series.parts],
                "target_audience": self.state.target_audience,
            }
        )
        self.state.validation_report = validation_output
        return validation_output

    @listen("validate_content")
    def maybe_improve_script(self):
        # If there are improvement suggestions and we haven't iterated yet, iterate once
        if self.state.validation_report and self.state.validation_report.improvement_suggestions and not self.state.iteration_done:
            print("Improvements suggested. Re-running scriptwriting and visual creation...")
            self.state.iteration_done = True
            # Re-run scriptwriting crew with feedback
            scriptwriting_crew = ScriptwritingCrew().crew()
            script_output = scriptwriting_crew.kickoff(
                inputs={
                    "research_parts": [part.dict() for part in self.state.research_series.parts],
                    "target_audience": self.state.target_audience,
                    "critic_feedback": self.state.validation_report.improvement_suggestions,
                }
            )
            self.state.script_series = script_output

            # Re-run visuals crew with updated scripts
            visual_crew = VisualCrew().crew()
            visuals_output = visual_crew.kickoff(
                inputs={
                    "scripts_parts": [script.dict() for script in self.state.script_series.parts],
                    "target_audience": self.state.target_audience,
                }
            )
            self.state.visuals_series = visuals_output
            # End after second iteration
            print("Flow completed after improvements.")
        else:
            print("No further improvements needed. Flow completed.")

def kickoff():
    script_flow = ScriptFlow()
    script_flow.kickoff()

def plot():
    script_flow = ScriptFlow()
    script_flow.plot()

if __name__ == "__main__":
    kickoff()