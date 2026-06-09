import 'package:flutter/material.dart';

import 'api_client.dart';
import 'models.dart';

void main() {
  runApp(const ProactiveAssistantApp());
}

class ProactiveAssistantApp extends StatelessWidget {
  const ProactiveAssistantApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Proactive Assistant',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xff256f5c)),
        useMaterial3: true,
        visualDensity: VisualDensity.compact,
      ),
      home: const SessionWorkspace(),
    );
  }
}

class SessionWorkspace extends StatefulWidget {
  const SessionWorkspace({super.key});

  @override
  State<SessionWorkspace> createState() => _SessionWorkspaceState();
}

class _SessionWorkspaceState extends State<SessionWorkspace> {
  final _baseUrl = TextEditingController(text: 'http://127.0.0.1:8001');
  final _sessionTitle = TextEditingController(text: 'Launch risk sync');
  final _speaker = TextEditingController(text: 'Bao');
  final _transcriptText = TextEditingController(
    text: '这个问题谁负责，下周五 deadline 前能不能定？',
  );

  AssistantSession? _session;
  SessionState? _state;
  SummaryResult? _summary;
  bool _loading = false;
  String _status = 'Ready';
  int _clockMs = 0;

  ProactiveApiClient get _api => ProactiveApiClient(baseUrl: _baseUrl.text.trim());

  @override
  void dispose() {
    _baseUrl.dispose();
    _sessionTitle.dispose();
    _speaker.dispose();
    _transcriptText.dispose();
    super.dispose();
  }

  Future<void> _run(Future<void> Function() action) async {
    setState(() {
      _loading = true;
      _status = 'Working';
    });
    try {
      await action();
    } on Object catch (error) {
      setState(() => _status = error.toString());
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  Future<void> _createSession() {
    return _run(() async {
      final session = await _api.createSession(title: _sessionTitle.text.trim());
      final state = await _api.getSessionState(session.sessionId);
      setState(() {
        _session = session;
        _state = state;
        _summary = null;
        _clockMs = 0;
        _status = 'Session created';
      });
    });
  }

  Future<void> _refresh() {
    final session = _session;
    if (session == null) {
      return _createSession();
    }
    return _run(() async {
      final state = await _api.getSessionState(session.sessionId);
      setState(() {
        _session = state.session;
        _state = state;
        _status = 'Refreshed';
      });
    });
  }

  Future<void> _appendTranscript() {
    final session = _session;
    if (session == null) {
      return _createSession().then((_) => _appendTranscript());
    }
    final text = _transcriptText.text.trim();
    if (text.isEmpty) {
      setState(() => _status = 'Transcript is empty');
      return Future.value();
    }
    return _run(() async {
      final startMs = _clockMs;
      final endMs = startMs + 900;
      await _api.appendTranscript(
        sessionId: session.sessionId,
        speaker: _speaker.text.trim().isEmpty ? 'unknown' : _speaker.text.trim(),
        text: text,
        startMs: startMs,
        endMs: endMs,
      );
      final state = await _api.getSessionState(session.sessionId);
      setState(() {
        _clockMs = endMs + 100;
        _session = state.session;
        _state = state;
        _status = 'Transcript appended';
      });
    });
  }

  Future<void> _feedback(PromptPayload prompt, FeedbackSignal signal) {
    return _run(() async {
      await _api.recordFeedback(decisionId: prompt.decisionId, signal: signal);
      if (_session != null) {
        final state = await _api.getSessionState(_session!.sessionId);
        setState(() => _state = state);
      }
      setState(() => _status = 'Feedback recorded: ${signal.value}');
    });
  }

  Future<void> _generateSummary() {
    final session = _session;
    if (session == null) {
      setState(() => _status = 'Create a session first');
      return Future.value();
    }
    return _run(() async {
      final summary = await _api.generateSummary(session.sessionId);
      final state = await _api.getSessionState(session.sessionId);
      setState(() {
        _summary = summary;
        _state = state;
        _status = 'Summary generated';
      });
    });
  }

  Future<void> _endSession() {
    final session = _session;
    if (session == null) {
      setState(() => _status = 'Create a session first');
      return Future.value();
    }
    return _run(() async {
      final summary = await _api.endSession(session.sessionId);
      final state = await _api.getSessionState(session.sessionId);
      setState(() {
        _summary = summary;
        _session = state.session;
        _state = state;
        _status = 'Session ended';
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    final state = _state;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Proactive Assistant'),
        actions: [
          IconButton(
            tooltip: 'Refresh',
            onPressed: _loading ? null : _refresh,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            final wide = constraints.maxWidth >= 900;
            final controls = _ControlsPanel(
              baseUrl: _baseUrl,
              sessionTitle: _sessionTitle,
              speaker: _speaker,
              transcriptText: _transcriptText,
              session: _session,
              loading: _loading,
              status: _status,
              onCreateSession: _createSession,
              onAppendTranscript: _appendTranscript,
              onGenerateSummary: _generateSummary,
              onEndSession: _endSession,
            );
            final content = _WorkspaceContent(
              state: state,
              summary: _summary,
              onFeedback: _feedback,
            );
            if (wide) {
              return Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  SizedBox(
                    width: 360,
                    child: SingleChildScrollView(child: controls),
                  ),
                  const VerticalDivider(width: 1),
                  Expanded(child: content),
                ],
              );
            }
            return ListView(
              children: [
                controls,
                const Divider(height: 1),
                SizedBox(height: 680, child: content),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _ControlsPanel extends StatelessWidget {
  const _ControlsPanel({
    required this.baseUrl,
    required this.sessionTitle,
    required this.speaker,
    required this.transcriptText,
    required this.session,
    required this.loading,
    required this.status,
    required this.onCreateSession,
    required this.onAppendTranscript,
    required this.onGenerateSummary,
    required this.onEndSession,
  });

  final TextEditingController baseUrl;
  final TextEditingController sessionTitle;
  final TextEditingController speaker;
  final TextEditingController transcriptText;
  final AssistantSession? session;
  final bool loading;
  final String status;
  final VoidCallback onCreateSession;
  final VoidCallback onAppendTranscript;
  final VoidCallback onGenerateSummary;
  final VoidCallback onEndSession;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _StatusStrip(session: session, loading: loading, status: status),
          const SizedBox(height: 16),
          TextField(
            controller: baseUrl,
            decoration: const InputDecoration(
              labelText: 'API Base URL',
              prefixIcon: Icon(Icons.dns_outlined),
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: sessionTitle,
            decoration: const InputDecoration(
              labelText: 'Session Title',
              prefixIcon: Icon(Icons.event_note_outlined),
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: loading ? null : onCreateSession,
            icon: const Icon(Icons.add_circle_outline),
            label: const Text('Create Session'),
          ),
          const SizedBox(height: 20),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: speaker,
                  decoration: const InputDecoration(
                    labelText: 'Speaker',
                    prefixIcon: Icon(Icons.person_outline),
                    border: OutlineInputBorder(),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          TextField(
            controller: transcriptText,
            minLines: 5,
            maxLines: 8,
            decoration: const InputDecoration(
              labelText: 'Transcript',
              alignLabelWithHint: true,
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: loading ? null : onAppendTranscript,
            icon: const Icon(Icons.playlist_add),
            label: const Text('Append Transcript'),
          ),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            onPressed: loading ? null : onGenerateSummary,
            icon: const Icon(Icons.summarize_outlined),
            label: const Text('Generate Summary'),
          ),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            onPressed: loading ? null : onEndSession,
            icon: const Icon(Icons.stop_circle_outlined),
            label: const Text('End Session'),
          ),
        ],
      ),
    );
  }
}

class _StatusStrip extends StatelessWidget {
  const _StatusStrip({
    required this.session,
    required this.loading,
    required this.status,
  });

  final AssistantSession? session;
  final bool loading;
  final String status;

  @override
  Widget build(BuildContext context) {
    final color = Theme.of(context).colorScheme;
    return DecoratedBox(
      decoration: BoxDecoration(
        color: color.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  loading ? Icons.sync : Icons.radio_button_checked,
                  size: 18,
                  color: loading ? color.primary : color.tertiary,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    status,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ),
              ],
            ),
            if (session != null) ...[
              const SizedBox(height: 8),
              Text(
                '${session!.sessionId} · ${session!.status}',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: Theme.of(context).textTheme.labelMedium,
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _WorkspaceContent extends StatelessWidget {
  const _WorkspaceContent({
    required this.state,
    required this.summary,
    required this.onFeedback,
  });

  final SessionState? state;
  final SummaryResult? summary;
  final Future<void> Function(PromptPayload prompt, FeedbackSignal signal) onFeedback;

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 3,
      child: Column(
        children: [
          const TabBar(
            tabs: [
              Tab(icon: Icon(Icons.subject), text: 'Transcript'),
              Tab(icon: Icon(Icons.tips_and_updates_outlined), text: 'Prompts'),
              Tab(icon: Icon(Icons.fact_check_outlined), text: 'Summary'),
            ],
          ),
          Expanded(
            child: TabBarView(
              children: [
                _TranscriptTab(state: state),
                _PromptsTab(state: state, onFeedback: onFeedback),
                _SummaryTab(state: state, summary: summary),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _TranscriptTab extends StatelessWidget {
  const _TranscriptTab({required this.state});

  final SessionState? state;

  @override
  Widget build(BuildContext context) {
    final segments = state?.transcript ?? const <TranscriptSegment>[];
    if (segments.isEmpty) {
      return const _EmptyState(icon: Icons.subject, label: 'No transcript');
    }
    return ListView.separated(
      padding: const EdgeInsets.all(16),
      itemCount: segments.length + 1,
      separatorBuilder: (_, __) => const SizedBox(height: 8),
      itemBuilder: (context, index) {
        if (index == 0) {
          return _MeetingStats(state: state!);
        }
        final segment = segments[index - 1];
        return ListTile(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          tileColor: Theme.of(context).colorScheme.surfaceContainerHighest,
          leading: const Icon(Icons.record_voice_over_outlined),
          title: Text(segment.speaker),
          subtitle: Text(segment.text),
          trailing: Text('${segment.startMs}ms'),
        );
      },
    );
  }
}

class _MeetingStats extends StatelessWidget {
  const _MeetingStats({required this.state});

  final SessionState state;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        _MetricChip(icon: Icons.task_alt, label: 'Actions', value: state.actionItemCount),
        _MetricChip(icon: Icons.help_outline, label: 'Questions', value: state.openQuestionCount),
        _MetricChip(icon: Icons.warning_amber_outlined, label: 'Risks', value: state.riskCount),
      ],
    );
  }
}

class _MetricChip extends StatelessWidget {
  const _MetricChip({
    required this.icon,
    required this.label,
    required this.value,
  });

  final IconData icon;
  final String label;
  final int value;

  @override
  Widget build(BuildContext context) {
    return Chip(
      avatar: Icon(icon, size: 18),
      label: Text('$label $value'),
    );
  }
}

class _PromptsTab extends StatelessWidget {
  const _PromptsTab({
    required this.state,
    required this.onFeedback,
  });

  final SessionState? state;
  final Future<void> Function(PromptPayload prompt, FeedbackSignal signal) onFeedback;

  @override
  Widget build(BuildContext context) {
    final prompts = state?.prompts ?? const <PromptPayload>[];
    if (prompts.isEmpty) {
      return const _EmptyState(icon: Icons.tips_and_updates_outlined, label: 'No prompts');
    }
    return ListView.separated(
      padding: const EdgeInsets.all(16),
      itemCount: prompts.length,
      separatorBuilder: (_, __) => const SizedBox(height: 12),
      itemBuilder: (context, index) {
        return _PromptCard(prompt: prompts[index], onFeedback: onFeedback);
      },
    );
  }
}

class _PromptCard extends StatelessWidget {
  const _PromptCard({
    required this.prompt,
    required this.onFeedback,
  });

  final PromptPayload prompt;
  final Future<void> Function(PromptPayload prompt, FeedbackSignal signal) onFeedback;

  @override
  Widget build(BuildContext context) {
    final color = Theme.of(context).colorScheme;
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
        side: BorderSide(color: color.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.visibility_outlined, color: color.primary),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    prompt.glassesTitle.isEmpty ? prompt.promptCategory : prompt.glassesTitle,
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
                Chip(label: Text('G${prompt.contentGranularity}')),
              ],
            ),
            if (prompt.glassesText.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(prompt.glassesText),
            ],
            if (prompt.appDetailText.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(
                prompt.appDetailText,
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                IconButton.filledTonal(
                  tooltip: 'Accept',
                  onPressed: () => onFeedback(prompt, FeedbackSignal.nodAccept),
                  icon: const Icon(Icons.thumb_up_alt_outlined),
                ),
                IconButton.filledTonal(
                  tooltip: 'Reject',
                  onPressed: () => onFeedback(prompt, FeedbackSignal.headShakeReject),
                  icon: const Icon(Icons.thumb_down_alt_outlined),
                ),
                IconButton.filledTonal(
                  tooltip: 'Open detail',
                  onPressed: () => onFeedback(prompt, FeedbackSignal.openDetail),
                  icon: const Icon(Icons.open_in_new),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _SummaryTab extends StatelessWidget {
  const _SummaryTab({
    required this.state,
    required this.summary,
  });

  final SessionState? state;
  final SummaryResult? summary;

  @override
  Widget build(BuildContext context) {
    final prompts = summary?.prompts ??
        (state?.prompts.where((prompt) => prompt.prdSurface == 'app_summary_tab').toList() ?? const <PromptPayload>[]);
    if (prompts.isEmpty) {
      return const _EmptyState(icon: Icons.fact_check_outlined, label: 'No summary');
    }
    return ListView.separated(
      padding: const EdgeInsets.all(16),
      itemCount: prompts.length,
      separatorBuilder: (_, __) => const SizedBox(height: 12),
      itemBuilder: (context, index) {
        final prompt = prompts[index];
        return ListTile(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          tileColor: Theme.of(context).colorScheme.surfaceContainerHighest,
          leading: const Icon(Icons.summarize_outlined),
          title: Text(prompt.glassesTitle),
          subtitle: Text(prompt.appDetailText.isEmpty ? prompt.glassesText : prompt.appDetailText),
        );
      },
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState({
    required this.icon,
    required this.label,
  });

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 42, color: Theme.of(context).colorScheme.outline),
          const SizedBox(height: 8),
          Text(label),
        ],
      ),
    );
  }
}
