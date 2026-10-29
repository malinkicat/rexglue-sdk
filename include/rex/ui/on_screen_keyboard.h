/**
 * @file        rex/ui/on_screen_keyboard.h
 * @brief       Controller-driven on-screen keyboard (model only, no drawing)
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rex::ui {

// A QWERTY grid of 10 columns: a number row, three letter rows (with ' - . !)
// and a bottom row of wide keys (Shift, Space, Backspace, OK). The highlight
// moves with the D-pad/stick, wrapping at the edges, and keeps its column when
// moving between rows.
class OnScreenKeyboard {
 public:
  enum class KeyType : uint8_t { kChar, kShift, kSpace, kBackspace, kOk };
  struct Key {
    KeyType type;
    char ch;   // kChar: the character (upper case for letters)
    int span;  // columns of the 10-column grid
  };
  enum class Result { kNone, kDone, kCancel };

  OnScreenKeyboard(std::string text, size_t max_length);

  static const std::vector<std::vector<Key>>& Layout();

  void Move(int dx, int dy);
  // A on the highlighted key.
  Result Press();
  // B: deletes the last character, or cancels when the text is empty.
  Result Back();
  // X.
  void Space();
  // Typed (also from a physical keyboard): false if the keyboard doesn't
  // have the character or the text is at the maximum length.
  bool TypeChar(char c);
  void Backspace();

  const std::string& text() const { return text_; }
  size_t row() const { return row_; }
  size_t key() const { return key_; }
  bool upper() const { return upper_; }

 private:
  static int KeyColumn(size_t row, size_t key);
  static size_t KeyAtColumn(size_t row, int column);

  std::string text_;
  size_t max_length_;
  size_t row_ = 0;
  size_t key_ = 0;
  int column_ = 0;  // preferred column for moving between rows
  bool upper_ = true;
};

}  // namespace rex::ui
