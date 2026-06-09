enum FeedbackSignal {
  nodAccept('nod_accept'),
  headShakeReject('head_shake_reject'),
  manualRequest('manual_request'),
  openDetail('open_detail');

  const FeedbackSignal(this.value);
  final String value;
}

class AssistantSession {
  const AssistantSession({
    required this.sessionId,
    required this.status,
    required this.title,
  });

  factory AssistantSession.fromJson(Map<String, dynamic> json) {
    return AssistantSession(
      sessionId: json['session_id'] as String? ?? '',
      status: json['status'] as String? ?? '',
      title: json['title'] as String? ?? '',
    );
  }

  final String sessionId;
  final String status;
  final String title;
}

class TranscriptSegment {
  const TranscriptSegment({
    required this.segmentId,
    required this.speaker,
    required this.text,
    required this.startMs,
    required this.endMs,
  });

  factory TranscriptSegment.fromJson(Map<String, dynamic> json) {
    return TranscriptSegment(
      segmentId: json['segment_id'] as String? ?? '',
      speaker: json['speaker'] as String? ?? '',
      text: json['text'] as String? ?? '',
      startMs: json['start_ms'] as int? ?? 0,
      endMs: json['end_ms'] as int? ?? 0,
    );
  }

  final String segmentId;
  final String speaker;
  final String text;
  final int startMs;
  final int endMs;
}

class PromptPayload {
  const PromptPayload({
    required this.decisionId,
    required this.shouldDisplay,
    required this.displayStatus,
    required this.promptCategory,
    required this.contentGranularity,
    required this.prdSurface,
    required this.glassesTitle,
    required this.glassesText,
    required this.appDetailText,
    required this.confidence,
  });

  factory PromptPayload.fromJson(Map<String, dynamic> json) {
    return PromptPayload(
      decisionId: json['decision_id'] as String? ?? '',
      shouldDisplay: json['should_display'] as bool? ?? false,
      displayStatus: json['display_status'] as String? ?? '',
      promptCategory: json['prompt_category'] as String? ?? '',
      contentGranularity: json['content_granularity'] as int? ?? 0,
      prdSurface: json['prd_surface'] as String? ?? '',
      glassesTitle: json['glasses_title'] as String? ?? '',
      glassesText: json['glasses_text'] as String? ?? '',
      appDetailText: json['app_detail_text'] as String? ?? '',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0,
    );
  }

  final String decisionId;
  final bool shouldDisplay;
  final String displayStatus;
  final String promptCategory;
  final int contentGranularity;
  final String prdSurface;
  final String glassesTitle;
  final String glassesText;
  final String appDetailText;
  final double confidence;
}

class SessionState {
  const SessionState({
    required this.session,
    required this.transcript,
    required this.prompts,
    required this.actionItemCount,
    required this.openQuestionCount,
    required this.riskCount,
  });

  factory SessionState.fromJson(Map<String, dynamic> json) {
    final meetingState = json['meeting_state'] as Map<String, dynamic>? ?? {};
    return SessionState(
      session: AssistantSession.fromJson(json['session'] as Map<String, dynamic>? ?? {}),
      transcript: _list(json['transcript']).map(TranscriptSegment.fromJson).toList(),
      prompts: _list(json['prompts']).map(PromptPayload.fromJson).toList(),
      actionItemCount: _list(meetingState['action_items']).length,
      openQuestionCount: _list(meetingState['open_questions']).length,
      riskCount: _list(meetingState['risks']).length,
    );
  }

  final AssistantSession session;
  final List<TranscriptSegment> transcript;
  final List<PromptPayload> prompts;
  final int actionItemCount;
  final int openQuestionCount;
  final int riskCount;
}

class TranscriptStepResult {
  const TranscriptStepResult({
    required this.session,
    required this.transcriptSegment,
    required this.prompts,
  });

  factory TranscriptStepResult.fromJson(Map<String, dynamic> json) {
    return TranscriptStepResult(
      session: AssistantSession.fromJson(json['session'] as Map<String, dynamic>? ?? {}),
      transcriptSegment: TranscriptSegment.fromJson(
        json['transcript_segment'] as Map<String, dynamic>? ?? {},
      ),
      prompts: _list(json['prompts']).map(PromptPayload.fromJson).toList(),
    );
  }

  final AssistantSession session;
  final TranscriptSegment transcriptSegment;
  final List<PromptPayload> prompts;
}

class SummaryResult {
  const SummaryResult({required this.prompts});

  factory SummaryResult.fromJson(Map<String, dynamic> json) {
    return SummaryResult(
      prompts: _list(json['prompts']).map(PromptPayload.fromJson).toList(),
    );
  }

  final List<PromptPayload> prompts;
}

List<Map<String, dynamic>> _list(Object? value) {
  if (value is! List) {
    return const [];
  }
  return value.whereType<Map<String, dynamic>>().toList();
}
