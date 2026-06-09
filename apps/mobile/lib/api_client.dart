import 'dart:convert';

import 'package:http/http.dart' as http;

import 'models.dart';

class ApiException implements Exception {
  const ApiException(this.message);

  final String message;

  @override
  String toString() => message;
}

class ProactiveApiClient {
  ProactiveApiClient({required this.baseUrl, http.Client? httpClient})
      : _httpClient = httpClient;

  final String baseUrl;
  final http.Client? _httpClient;

  Uri _uri(String path, [Map<String, String>? query]) {
    final root = baseUrl.endsWith('/') ? baseUrl.substring(0, baseUrl.length - 1) : baseUrl;
    final uri = Uri.parse('$root$path');
    return query == null ? uri : uri.replace(queryParameters: query);
  }

  Future<AssistantSession> createSession({required String title}) async {
    final payload = await _postJson('/sessions', {
      'config': {
        'title': title,
        'source': 'manual_transcript',
        'locale': 'zh-CN',
      },
    });
    return AssistantSession.fromJson(payload['session'] as Map<String, dynamic>? ?? {});
  }

  Future<SessionState> getSessionState(String sessionId) async {
    final payload = await _getJson('/sessions/$sessionId/state');
    return SessionState.fromJson(payload);
  }

  Future<TranscriptStepResult> appendTranscript({
    required String sessionId,
    required String speaker,
    required String text,
    required int startMs,
    required int endMs,
  }) async {
    final payload = await _postJson('/sessions/$sessionId/transcript', {
      'segment': {
        'speaker': speaker,
        'start_ms': startMs,
        'end_ms': endMs,
        'text': text,
        'asr_confidence': 1.0,
        'source': 'manual_transcript',
      },
    });
    return TranscriptStepResult.fromJson(payload);
  }

  Future<void> recordFeedback({
    required String decisionId,
    required FeedbackSignal signal,
  }) async {
    await _postJson('/prompt-decisions/$decisionId/feedback', {
      'signal_type': signal.value,
      'propose_memory': false,
    });
  }

  Future<SummaryResult> generateSummary(String sessionId) async {
    final payload = await _postJson('/sessions/$sessionId/summary', {
      'use_memory': true,
      'include_pending_memory': true,
    });
    return SummaryResult.fromJson(payload);
  }

  Future<SummaryResult?> endSession(String sessionId, {bool generateSummary = true}) async {
    final payload = await _postJson('/sessions/$sessionId/end', {
      'generate_summary': generateSummary,
      'use_memory': true,
      'include_pending_memory': true,
    });
    final summary = payload['summary'];
    if (summary is Map<String, dynamic>) {
      return SummaryResult.fromJson(summary);
    }
    return null;
  }

  Future<Map<String, dynamic>> _getJson(String path) async {
    final uri = _uri(path);
    final client = _httpClient;
    final response = client == null ? await http.get(uri) : await client.get(uri);
    return _decodeResponse(response);
  }

  Future<Map<String, dynamic>> _postJson(String path, Map<String, dynamic> body) async {
    final uri = _uri(path);
    final headers = {'content-type': 'application/json'};
    final encodedBody = jsonEncode(body);
    final client = _httpClient;
    final response = client == null
        ? await http.post(uri, headers: headers, body: encodedBody)
        : await client.post(uri, headers: headers, body: encodedBody);
    return _decodeResponse(response);
  }

  Map<String, dynamic> _decodeResponse(http.Response response) {
    final decoded = response.body.isEmpty ? <String, dynamic>{} : jsonDecode(response.body);
    if (response.statusCode < 200 || response.statusCode >= 300) {
      final detail = decoded is Map<String, dynamic> ? decoded['detail'] : null;
      throw ApiException(detail?.toString() ?? 'HTTP ${response.statusCode}');
    }
    if (decoded is Map<String, dynamic>) {
      return decoded;
    }
    throw const ApiException('Backend returned non-object JSON');
  }
}
