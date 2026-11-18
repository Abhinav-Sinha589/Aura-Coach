import { countFillers, createSpeechTracker, computeWpm, normalizeTranscript } from "./speech";

// Build SpeechRecognitionResultList-like objects
const res = (text, isFinal) => Object.assign([{ transcript: text }], { isFinal });
const list = (...items) => items;

describe("filler detection", () => {
  test("counts single and phrase fillers with punctuation", () => {
    expect(countFillers("Um, so, you know, I mean... uh it's kind of hard.")).toBe(5);
  });
  test("phrase tokens are not double counted", () => {
    expect(countFillers("you know you know")).toBe(2);
  });
  test("'like' is only a filler in filler contexts", () => {
    expect(countFillers("I like pizza and it looks like rain")).toBe(0);
    expect(countFillers("and like, um, it was like so like great")).toBe(3);
  });
  test("literal 'a kind of' is not a filler", () => {
    expect(countFillers("it is a kind of fruit")).toBe(0);
  });
  test("normalizes curly apostrophes and case", () => {
    expect(normalizeTranscript("It’s  UM  Fine!")).toBe("it's um fine");
  });
});

describe("speech tracker", () => {
  test("interim results are never counted", () => {
    const t = createSpeechTracker();
    t.ingest(list(res("hello", false)), 0);
    t.ingest(list(res("hello there", false)), 0);
    t.ingest(list(res("hello there friend", false)), 0);
    expect(t.words).toBe(0);
  });

  test("a final result counts once even if the event repeats", () => {
    const t = createSpeechTracker();
    const results = list(res("hello there friend", true));
    t.ingest(results, 0);
    t.ingest(results, 0); // Chrome re-delivers earlier finals in later events
    expect(t.words).toBe(3);
  });

  test("later events with resultIndex only add the new segment", () => {
    const t = createSpeechTracker();
    t.ingest(list(res("one two", true)), 0, 0);
    t.ingest(list(res("one two", true), res("three four five", true)), 1, 5000);
    expect(t.words).toBe(5);
  });

  test("restart (new generation) after a pause keeps counting", () => {
    const t = createSpeechTracker();
    t.ingest(list(res("first part", true)), 0, 0);
    t.newGeneration(); // recognition ended on a pause and restarted: indices reset
    t.ingest(list(res("second part here", true)), 0, 10000);
    expect(t.words).toBe(5);
  });

  test("growing duplicate finals (Android) count only the delta", () => {
    const t = createSpeechTracker();
    t.ingest(list(res("hello", true)), 0, 1000);
    t.ingest(list(res("hello", true), res("hello world", true)), 1, 1500);
    expect(t.words).toBe(2);
  });

  test("fillers accumulate and recent window expires", () => {
    const t = createSpeechTracker();
    t.ingest(list(res("um uh you know", true)), 0, 0);
    expect(t.fillers).toBe(3);
    expect(t.recentFillers(30000)).toBe(3);
    expect(t.recentFillers(61000)).toBe(0);
    expect(t.fillers).toBe(3);
  });

  test("no 1000+ WPM from duplicate events", () => {
    const t = createSpeechTracker();
    const results = [];
    for (let i = 0; i < 20; i++) {
      results.push(res("this is a short sentence", true));
      // the whole growing list is re-delivered on every event
      t.ingest(results, 0, i * 3000);
      t.ingest(results, 0, i * 3000 + 10);
    }
    expect(t.words).toBe(100);
    expect(computeWpm(t.words, 60000)).toBe(100);
  });
});

describe("computeWpm", () => {
  test("null until enough time has elapsed", () => {
    expect(computeWpm(10, 2000)).toBeNull();
    expect(computeWpm(150, 60000)).toBe(150);
    expect(computeWpm(75, 30000)).toBe(150);
  });
});
