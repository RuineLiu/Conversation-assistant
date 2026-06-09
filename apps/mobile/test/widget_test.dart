import 'package:flutter_test/flutter_test.dart';

import 'package:proactive_assistant_mobile/main.dart';

void main() {
  testWidgets('renders proactive assistant workspace', (WidgetTester tester) async {
    await tester.pumpWidget(const ProactiveAssistantApp());

    expect(find.text('Proactive Assistant'), findsOneWidget);
    expect(find.text('Create Session'), findsOneWidget);
    expect(find.text('Append Transcript'), findsOneWidget);
  });
}
