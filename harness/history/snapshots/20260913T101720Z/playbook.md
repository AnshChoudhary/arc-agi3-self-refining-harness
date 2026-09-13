# Playbook (v0, hand-written)

Ordered rules. Earlier rules win when two conflict.

1. Every environment action costs score; analysis code is free. Run analysis until you have a concrete hypothesis before acting.
2. Right after a level starts, probe each available action once (for click actions, click one distinct small object), reading the diff after each. Record what every action does in your notes before trying to make progress.
3. An action that changed nothing twice in a row is a no-op in that state. Stop repeating it.
4. Keep a running model in your notes: what each action does, which objects move, what looks like a goal or a counter, what you have ruled out. Rewrite the notes every action; they are your only memory beyond the last few steps.
5. Small components that differ from the background are usually interactive or informative: agents, targets, counters, indicators. Look at them first with `show()`.
6. For click actions, aim at the centre of a component, not at empty background. Try a different component if a click changed nothing.
7. Watch for a level counter, a lives counter, or a shrinking bar at the frame edge; it tells you what the game rewards or punishes.
8. If undo (ACTION7) is available it still costs an action; only use it to escape a state you know is bad.
9. When the level index increases, the mechanics usually carry over. Reuse your notes and re-probe only what looks new.
10. If the state is GAME_OVER the loop resets the level for you; treat the sequence that led there as a hazard and write it down.
