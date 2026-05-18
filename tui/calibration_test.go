// Tests for the calibration badge renderer (PC-059).

package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRenderCalibrationBadge_NilStatus_ShowsPlaceholder(t *testing.T) {
	got := renderCalibrationBadge(nil)
	if !strings.Contains(got, "cal") {
		t.Errorf("expected 'cal' placeholder, got %q", got)
	}
}

func TestRenderCalibrationBadge_BothSupported_ShowsCheck(t *testing.T) {
	s := &calibrationStatus{}
	s.Lens.Verdict = "supported"
	s.ASA.Verdict = "supported"
	got := renderCalibrationBadge(s)
	// ANSI codes wrap each badge — assert on the visible text content
	if !strings.Contains(got, "Lens ✓") {
		t.Errorf("expected 'Lens ✓' badge, got %q", got)
	}
	if !strings.Contains(got, "ASA ✓") {
		t.Errorf("expected 'ASA ✓' badge, got %q", got)
	}
}

func TestRenderCalibrationBadge_NoArtifacts_ShowsWarning(t *testing.T) {
	s := &calibrationStatus{}
	s.Lens.Verdict = "no-artifacts"
	s.ASA.Verdict = "supported"
	got := renderCalibrationBadge(s)
	if !strings.Contains(got, "Lens ⚠") {
		t.Errorf("expected 'Lens ⚠' warn badge, got %q", got)
	}
	if !strings.Contains(got, "ASA ✓") {
		t.Errorf("expected 'ASA ✓' ok badge alongside warn, got %q", got)
	}
}

func TestRenderCalibrationBadge_Unreachable_ShowsFail(t *testing.T) {
	s := &calibrationStatus{}
	s.Lens.Verdict = "unreachable"
	s.ASA.Verdict = "missing"
	got := renderCalibrationBadge(s)
	if !strings.Contains(got, "Lens ✗") {
		t.Errorf("expected 'Lens ✗' fail badge, got %q", got)
	}
	if !strings.Contains(got, "ASA ⚠") {
		t.Errorf("expected 'ASA ⚠' warn badge for missing, got %q", got)
	}
}

func TestFetchCalibrationStatusCmd_RoundTripsRealResponse(t *testing.T) {
	// Spin up a mock proxy that returns a realistic CalibrationStatus
	// payload. The cmd should parse it into the right struct.
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/calibration/status", func(w http.ResponseWriter, r *http.Request) {
		payload := map[string]any{
			"lens": map[string]any{
				"verdict":           "supported",
				"cost_field_loaded": true,
				"cost_field_dim":    4096,
				"embed_dim":         4096,
				"gx_loaded":         true,
				"hint":              "ready",
			},
			"asa": map[string]any{
				"verdict":        "missing",
				"vector_path":    "/models/ast_edit_steering.gguf",
				"vector_present": false,
				"hint":           "no control vector",
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(payload)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	msg := fetchCalibrationStatusCmd(srv.URL)()
	cmsg, ok := msg.(calibrationStatusMsg)
	if !ok {
		t.Fatalf("expected calibrationStatusMsg, got %T", msg)
	}
	if cmsg.err != nil {
		t.Fatalf("unexpected err: %v", cmsg.err)
	}
	if cmsg.status == nil {
		t.Fatal("expected non-nil status")
	}
	if cmsg.status.Lens.Verdict != "supported" {
		t.Errorf("Lens.Verdict = %q, want supported", cmsg.status.Lens.Verdict)
	}
	if cmsg.status.Lens.CostFieldDim != 4096 {
		t.Errorf("CostFieldDim = %d, want 4096", cmsg.status.Lens.CostFieldDim)
	}
	if cmsg.status.ASA.Verdict != "missing" {
		t.Errorf("ASA.Verdict = %q, want missing", cmsg.status.ASA.Verdict)
	}
	if cmsg.status.ASA.VectorPresent {
		t.Errorf("VectorPresent = true, want false")
	}
}

func TestFetchCalibrationStatusCmd_ProxyDown_ReturnsErr(t *testing.T) {
	// Point at a definitely-closed port.
	msg := fetchCalibrationStatusCmd("http://127.0.0.1:1")()
	cmsg, ok := msg.(calibrationStatusMsg)
	if !ok {
		t.Fatalf("expected calibrationStatusMsg, got %T", msg)
	}
	if cmsg.err == nil {
		t.Error("expected err for unreachable proxy, got nil")
	}
	if cmsg.status != nil {
		t.Error("expected nil status on err")
	}
}

func TestCalibrationTooltip_OnlyShowsNonSupported(t *testing.T) {
	s := &calibrationStatus{}
	s.Lens.Verdict = "supported"
	s.Lens.Hint = "ready"
	s.ASA.Verdict = "missing"
	s.ASA.Hint = "no control vector at /models/ast_edit_steering.gguf"
	got := calibrationTooltip(s)
	if strings.Contains(got, "Lens:") {
		t.Errorf("supported Lens should not appear in tooltip: %q", got)
	}
	if !strings.Contains(got, "ASA:") {
		t.Errorf("missing ASA should appear: %q", got)
	}
}

func TestCalibrationTooltip_NilStatus_Empty(t *testing.T) {
	if got := calibrationTooltip(nil); got != "" {
		t.Errorf("expected empty tooltip for nil status, got %q", got)
	}
}

func TestBadgeActionHint_BothSupported_Empty(t *testing.T) {
	s := &calibrationStatus{}
	s.Lens.Verdict = "supported"
	s.ASA.Verdict = "supported"
	if got := badgeActionHint(s); got != "" {
		t.Errorf("expected no hint when both supported, got %q", got)
	}
}

func TestBadgeActionHint_LensWarn_PointsAtBuildAndDocs(t *testing.T) {
	s := &calibrationStatus{}
	s.Lens.Verdict = "no-artifacts"
	s.ASA.Verdict = "supported"
	got := badgeActionHint(s)
	if !strings.Contains(got, "atlas lens build") {
		t.Errorf("lens-warn hint should suggest `atlas lens build`, got %q", got)
	}
	if !strings.Contains(got, "PUBLISHING.md") {
		t.Errorf("hint should reference docs/PUBLISHING.md, got %q", got)
	}
	if strings.Contains(got, "atlas asa build") {
		t.Errorf("ASA was supported, asa-build should NOT appear: %q", got)
	}
}

func TestBadgeActionHint_ASAWarn_PointsAtASABuild(t *testing.T) {
	s := &calibrationStatus{}
	s.Lens.Verdict = "supported"
	s.ASA.Verdict = "missing"
	got := badgeActionHint(s)
	if !strings.Contains(got, "atlas asa build") {
		t.Errorf("asa-warn hint should suggest `atlas asa build`, got %q", got)
	}
	if strings.Contains(got, "atlas lens build") {
		t.Errorf("Lens was supported, lens-build should NOT appear: %q", got)
	}
}

func TestBadgeActionHint_BothWarn_SuggestsBoth(t *testing.T) {
	s := &calibrationStatus{}
	s.Lens.Verdict = "dim-mismatch"
	s.ASA.Verdict = "missing"
	got := badgeActionHint(s)
	if !strings.Contains(got, "atlas lens build") || !strings.Contains(got, "atlas asa build") {
		t.Errorf("both-warn hint should mention both commands, got %q", got)
	}
}

func TestBadgeActionHint_Unreachable_NoBuildSuggestion(t *testing.T) {
	// When lens/asa are unreachable the artifact is fine — the service is
	// down. Telling the user to "build" would be misleading.
	s := &calibrationStatus{}
	s.Lens.Verdict = "unreachable"
	s.ASA.Verdict = "incompatible"
	if got := badgeActionHint(s); got != "" {
		t.Errorf("expected no build-hint for service-down verdicts, got %q", got)
	}
}

func TestRenderCalibrationBadge_WarnIncludesHint(t *testing.T) {
	s := &calibrationStatus{}
	s.Lens.Verdict = "no-artifacts"
	s.ASA.Verdict = "supported"
	got := renderCalibrationBadge(s)
	if !strings.Contains(got, "atlas lens build") {
		t.Errorf("rendered badge should append actionable hint when lens warn, got %q", got)
	}
	if !strings.Contains(got, "PUBLISHING.md") {
		t.Errorf("rendered badge should reference docs, got %q", got)
	}
}
