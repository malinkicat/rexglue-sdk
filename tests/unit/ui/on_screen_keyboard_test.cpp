/**
 * @file        ui/on_screen_keyboard_test.cpp
 * @brief       Tests for the controller on-screen keyboard model
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <catch2/catch_test_macros.hpp>

#include <string>

#include <rex/ui/on_screen_keyboard.h>

using rex::ui::OnScreenKeyboard;
using KeyType = OnScreenKeyboard::KeyType;
using Result = OnScreenKeyboard::Result;

namespace {
// The highlighted key.
const OnScreenKeyboard::Key& Highlighted(const OnScreenKeyboard& keyboard) {
  return OnScreenKeyboard::Layout()[keyboard.row()][keyboard.key()];
}
// Moves the highlight to the character key `ch` (letters/digits/punctuation).
void HighlightChar(OnScreenKeyboard& keyboard, char ch) {
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 12; ++j) {
      if (Highlighted(keyboard).type == KeyType::kChar && Highlighted(keyboard).ch == ch) return;
      keyboard.Move(1, 0);
    }
    keyboard.Move(0, 1);
  }
  FAIL("key not found");
}
}  // namespace

TEST_CASE("layout is QWERTY with a bottom row of wide keys", "[on_screen_keyboard]") {
  const auto& layout = OnScreenKeyboard::Layout();
  REQUIRE(layout.size() == 5);
  std::string rows[4];
  for (int r = 0; r < 4; ++r) {
    REQUIRE(layout[r].size() == 10);
    for (const auto& key : layout[r]) rows[r] += key.ch;
  }
  CHECK(rows[0] == "1234567890");
  CHECK(rows[1] == "QWERTYUIOP");
  CHECK(rows[2] == "ASDFGHJKL'");
  CHECK(rows[3] == "ZXCVBNM-.!");
  REQUIRE(layout[4].size() == 4);
  CHECK(layout[4][0].type == KeyType::kShift);
  CHECK(layout[4][0].span == 2);
  CHECK(layout[4][1].type == KeyType::kSpace);
  CHECK(layout[4][1].span == 3);
  CHECK(layout[4][2].type == KeyType::kBackspace);
  CHECK(layout[4][2].span == 2);
  CHECK(layout[4][3].type == KeyType::kOk);
  CHECK(layout[4][3].span == 3);
}

TEST_CASE("moving wraps at every edge", "[on_screen_keyboard]") {
  OnScreenKeyboard keyboard("Rider", 15);
  REQUIRE(keyboard.row() == 0);
  REQUIRE(keyboard.key() == 0);
  keyboard.Move(-1, 0);
  CHECK(keyboard.key() == 9);
  keyboard.Move(1, 0);
  CHECK(keyboard.key() == 0);
  keyboard.Move(0, -1);
  CHECK(keyboard.row() == 4);
  keyboard.Move(0, 1);
  CHECK(keyboard.row() == 0);
}

TEST_CASE("moving between rows keeps the column", "[on_screen_keyboard]") {
  OnScreenKeyboard keyboard("", 15);
  keyboard.Move(0, 1);  // Q
  CHECK(Highlighted(keyboard).ch == 'Q');
  keyboard.Move(0, 1);
  CHECK(Highlighted(keyboard).ch == 'A');
  keyboard.Move(-1, 0);  // to column 9: '
  keyboard.Move(0, -1);
  CHECK(Highlighted(keyboard).ch == 'P');
  keyboard.Move(0, 3);  // P column 9 down three rows: OK (columns 7-9)
  CHECK(Highlighted(keyboard).type == KeyType::kOk);
}

TEST_CASE("moving up from the space bar lands on its first column", "[on_screen_keyboard]") {
  OnScreenKeyboard keyboard("", 15);
  keyboard.Move(0, -1);  // bottom row, Shift
  keyboard.Move(1, 0);   // Space (columns 2-4)
  REQUIRE(Highlighted(keyboard).type == KeyType::kSpace);
  keyboard.Move(0, -1);
  CHECK(Highlighted(keyboard).ch == 'C');
}

TEST_CASE("B deletes, and cancels when the text is empty", "[on_screen_keyboard]") {
  OnScreenKeyboard keyboard("Rider", 15);
  CHECK(keyboard.Back() == Result::kNone);
  CHECK(keyboard.text() == "Ride");
  OnScreenKeyboard empty("", 15);
  CHECK(empty.Back() == Result::kCancel);
}

TEST_CASE("pressing a letter types it in the current case", "[on_screen_keyboard]") {
  OnScreenKeyboard keyboard("Rider", 15);
  CHECK(keyboard.upper());
  HighlightChar(keyboard, 'N');
  CHECK(keyboard.Press() == Result::kNone);
  CHECK(keyboard.text() == "RiderN");
  // Shift, then N again: lower case.
  keyboard.Move(0, 1);  // N (row 3) -> bottom row
  while (Highlighted(keyboard).type != KeyType::kShift) keyboard.Move(1, 0);
  keyboard.Press();
  CHECK_FALSE(keyboard.upper());
  HighlightChar(keyboard, 'N');
  keyboard.Press();
  CHECK(keyboard.text() == "RiderNn");
}

TEST_CASE("nothing is added past the maximum length", "[on_screen_keyboard]") {
  OnScreenKeyboard keyboard("ABCDEFGHIJKLMNO", 15);
  CHECK_FALSE(keyboard.TypeChar('A'));
  keyboard.Space();
  CHECK(keyboard.text() == "ABCDEFGHIJKLMNO");
  HighlightChar(keyboard, 'Q');
  keyboard.Press();
  CHECK(keyboard.text() == "ABCDEFGHIJKLMNO");
}

TEST_CASE("typing filters characters the keyboard doesn't have", "[on_screen_keyboard]") {
  OnScreenKeyboard keyboard("", 15);
  CHECK_FALSE(keyboard.TypeChar('@'));
  CHECK(keyboard.TypeChar('z'));
  CHECK(keyboard.TypeChar(' '));
  CHECK(keyboard.TypeChar('7'));
  CHECK(keyboard.text() == "z 7");
  keyboard.Backspace();
  CHECK(keyboard.text() == "z ");
}

TEST_CASE("OK finishes", "[on_screen_keyboard]") {
  OnScreenKeyboard keyboard("Rider", 15);
  keyboard.Move(0, -1);
  while (Highlighted(keyboard).type != KeyType::kOk) keyboard.Move(1, 0);
  CHECK(keyboard.Press() == Result::kDone);
  CHECK(keyboard.text() == "Rider");
}
