package main

// Plan-progress reminder injection. May 10 2026.
//
// Long multi-file tasks (e.g. "redo all 10 templates to match a SaaS
// design") lose sight of the plan once conversation trimming kicks in.
// The plan is generated up front via /v3/plan and stashed on ctx.Plan,
// and PlanStepsSatisfied tracks which steps have been hit — but neither
// surfaces back to the model after the original plan-rendering message
// drops out of the trim window.
//
// Fix: at the START of each LLM call we render a compact plan-progress
// "[system note]: ..." line and prepend it to the messages slice
// passed to callLLMOnce. The note is EPHEMERAL — it's not appended to
// ctx.Messages, so it doesn't accumulate or get re-trimmed. Every
// turn, the model sees: "step 3 of 7 — currently working on edit
// templates/dashboard.html; done: index.html, contact.html; remaining:
// pricing.html, services.html, ...".
//
// Cost: ~150 chars per turn. Cheap compared to letting the model
// re-read all the templates to remember what's done.

import (
	"fmt"
	"strings"
)

// buildPlanReminder returns a one-line "[system note]" string with
// plan progress, or "" if no plan is active. The caller prepends this
// to the messages slice passed to a single LLM call — it's not added
// to ctx.Messages, so it doesn't bloat history.
func buildPlanReminder(ctx *AgentContext) string {
	if ctx.Plan == nil || len(ctx.Plan.Steps) == 0 {
		return ""
	}
	if ctx.PlanStepsSatisfied == nil {
		ctx.PlanStepsSatisfied = make([]bool, len(ctx.Plan.Steps))
	}

	total := len(ctx.Plan.Steps)
	doneCount := 0
	doneIDs := make([]string, 0, total)
	remainingIDs := make([]string, 0, total)
	var current *PlanStep

	for i := range ctx.Plan.Steps {
		step := &ctx.Plan.Steps[i]
		if i < len(ctx.PlanStepsSatisfied) && ctx.PlanStepsSatisfied[i] {
			doneCount++
			doneIDs = append(doneIDs, step.ID)
		} else {
			if current == nil {
				current = step
			}
			remainingIDs = append(remainingIDs, step.ID)
		}
	}

	if current == nil {
		// All steps satisfied — the model should be on the verify step
		// or about to emit done. Surface that explicitly.
		return fmt.Sprintf(
			"[system note]: plan complete (%d/%d steps satisfied). Verify your work via `%s` if you haven't already, then emit `done` with a summary of what landed.",
			doneCount, total, planVerifyHint(ctx.Plan))
	}

	doneFrag := "none yet"
	if len(doneIDs) > 0 {
		doneFrag = strings.Join(doneIDs, ", ")
	}
	return fmt.Sprintf(
		"[system note]: plan progress %d/%d — currently on step %q (%s %s). Done: %s. Remaining: %s. Stay on the current step until it's complete; don't jump ahead and don't re-explore finished work.",
		doneCount, total, current.ID, current.Action, current.Target,
		doneFrag, strings.Join(remainingIDs, ", "),
	)
}

func planVerifyHint(p *Plan) string {
	if p == nil || p.VerifyStep == "" {
		return "the appropriate test/curl/run command"
	}
	return p.VerifyStep
}
