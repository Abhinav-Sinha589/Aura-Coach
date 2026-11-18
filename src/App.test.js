import { render, screen } from "@testing-library/react";

// Heavy browser-only libraries are not needed to render the home screen.
jest.mock("face-api.js", () => ({ nets: {}, TinyFaceDetectorOptions: function TinyFaceDetectorOptions() {} }));
jest.mock("@mediapipe/pose", () => ({ Pose: function Pose() {}, VERSION: "test" }));
jest.mock("chart.js/auto", () => function Chart() {});

// eslint-disable-next-line import/first
import App from "./App";

test("renders the home page with training and history entry points", () => {
  render(<App />);
  expect(screen.getByText(/Start Training/i)).toBeInTheDocument();
  expect(screen.getByText(/Session History/i)).toBeInTheDocument();
});
