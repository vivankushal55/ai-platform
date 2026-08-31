from agent import app

# LangGraph can describe its own structure.
# First, a text version that prints right in the terminal:
print(app.get_graph().draw_ascii())

# Then save a proper PNG image to put in your README:
png = app.get_graph().draw_mermaid_png()
with open("agent_graph.png", "wb") as f:
    f.write(png)
print("\nSaved diagram to agent_graph.png")
