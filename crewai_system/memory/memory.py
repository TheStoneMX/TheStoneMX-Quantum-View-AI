from crewai import Crew
from crewai.memory import CrewMemory

memory = CrewMemory()

self.crew = Crew(
    agents=[self.agent, ...],
    tasks=[...],
    memory=memory
)
