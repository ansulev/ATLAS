package main

import (
	"strings"
	"testing"
)

// May 10 2026 BiasBusters #30 — locks the reasoning-repetition
// detector against regression. Prefix-match similarity over normalized
// reasoning openings; ≥2 consecutive identical openings triggers
// intervention. Single-turn repeats and prose-free turns must NOT fire.

func TestRecordReasoningTriggersOnConsecutiveRepeat(t *testing.T) {
	ctx := &AgentContext{}
	// Turn 1: first reasoning. No intervention.
	if msg, fired := recordReasoning(ctx, "Now I need to read the file to understand the structure."); fired || msg != "" {
		t.Fatalf("turn 1: expected no fire, got fired=%v msg=%q", fired, msg)
	}
	// Turn 2: same opening prefix. count=1 (not yet at threshold of 2).
	if msg, fired := recordReasoning(ctx, "Now I need to read the file to understand the structure."); fired || msg != "" {
		t.Fatalf("turn 2: expected no fire (count=1, threshold=2), got fired=%v msg=%q", fired, msg)
	}
	// Turn 3: same opening prefix again. count=2. FIRES.
	msg, fired := recordReasoning(ctx, "Now I need to read the file to understand the structure.")
	if !fired {
		t.Fatalf("turn 3: expected intervention, got no fire")
	}
	if !strings.Contains(msg, "Reasoning repetition") {
		t.Errorf("intervention message missing canonical prefix: %s", msg)
	}
	if !strings.Contains(msg, "3 consecutive turns") {
		t.Errorf("intervention should report 3 consecutive turns, got: %s", msg)
	}
}

func TestRecordReasoningResetOnDivergence(t *testing.T) {
	ctx := &AgentContext{}
	recordReasoning(ctx, "Now I need to read the file.")
	recordReasoning(ctx, "Now I need to read the file.")
	// Turn 3: model commits to a different thought — counter resets.
	if _, fired := recordReasoning(ctx, "I have the file content. Now let me write the new version."); fired {
		t.Error("divergent reasoning should reset the counter, no intervention expected")
	}
	if ctx.ConsecutiveReasoningRepeats != 0 {
		t.Errorf("counter should reset to 0 after divergence, got %d", ctx.ConsecutiveReasoningRepeats)
	}
	// Turn 4: similar to turn 3 (the new pattern). count=1.
	if _, fired := recordReasoning(ctx, "I have the file content. Now let me write the new version."); fired {
		t.Error("turn 4 should be count=1 (one repeat), no fire yet")
	}
	// Turn 5: third identical → FIRES.
	if _, fired := recordReasoning(ctx, "I have the file content. Now let me write the new version."); !fired {
		t.Error("turn 5 should fire (count=2 of new pattern)")
	}
}

func TestRecordReasoningIgnoresEmptyTurns(t *testing.T) {
	ctx := &AgentContext{}
	recordReasoning(ctx, "Now I need to read the file.")
	// Turn 2: empty reasoning (model committed straight to action). Counter resets.
	if _, fired := recordReasoning(ctx, ""); fired {
		t.Error("empty reasoning should not fire")
	}
	if ctx.ConsecutiveReasoningRepeats != 0 || ctx.LastReasoningSnippet != "" {
		t.Errorf("empty reasoning should reset state; got repeats=%d snippet=%q",
			ctx.ConsecutiveReasoningRepeats, ctx.LastReasoningSnippet)
	}
	// Turn 3: same prose as turn 1, but the empty turn 2 broke the streak.
	// Should be treated as a fresh start, not count=2.
	recordReasoning(ctx, "Now I need to read the file.")
	if ctx.ConsecutiveReasoningRepeats != 0 {
		t.Errorf("post-empty: counter should be 0 (fresh start), got %d", ctx.ConsecutiveReasoningRepeats)
	}
}

func TestRecordReasoningNormalizesWhitespace(t *testing.T) {
	ctx := &AgentContext{}
	recordReasoning(ctx, "  Now I  need\nto    read the file.\n")
	recordReasoning(ctx, "now i need to read the file.")
	// Both should normalize to the same prefix → count=1.
	if ctx.ConsecutiveReasoningRepeats != 1 {
		t.Errorf("normalized whitespace+case should match; got count=%d, snippet=%q",
			ctx.ConsecutiveReasoningRepeats, ctx.LastReasoningSnippet)
	}
}

func TestRecordReasoningRespectsPrefixLength(t *testing.T) {
	// Two reasonings that share the first 80 chars but diverge later
	// should still match — that's the design (we want the OPENING to
	// be the signal). Let me confirm the prefix-match behavior.
	a := "Looking at the existing dashboard.html, I see the basic Flask template that needs to be transformed into a metrics view."
	b := "Looking at the existing dashboard.html, I see the basic Flask template that needs to be expanded with three KPI cards."
	ctx := &AgentContext{}
	recordReasoning(ctx, a)
	recordReasoning(ctx, b)
	// First 80 chars match → count should advance.
	if ctx.ConsecutiveReasoningRepeats == 0 {
		t.Errorf("expected prefix match to advance counter; got count=0, snippet=%q",
			ctx.LastReasoningSnippet)
	}
}

func TestRecordReasoningDoesNotFireOnSingleRepeat(t *testing.T) {
	ctx := &AgentContext{}
	recordReasoning(ctx, "Looking at the file...")
	if _, fired := recordReasoning(ctx, "Looking at the file..."); fired {
		t.Error("single repeat (turn 2 = turn 1) should not fire — needs 2 consecutive repeats")
	}
}

func TestNormalizeReasoningSnippet(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", ""},
		{"   \n  ", ""},
		{"Hello World", "hello world"},
		{"  HELLO\n\tWORLD  ", "hello world"},
		{strings.Repeat("a", 200), strings.Repeat("a", 80)},
	}
	for _, tc := range cases {
		if got := normalizeReasoningSnippet(tc.in); got != tc.want {
			t.Errorf("normalize(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
