/**
 * @file        ui/on_screen_keyboard.cpp
 * @brief       Controller-driven on-screen keyboard (model only, no drawing)
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <rex/ui/on_screen_keyboard.h>

#include <cctype>
#include <cstring>

namespace rex::ui {

namespace {
std::vector<OnScreenKeyboard::Key> CharRow(const char* chars) {
  std::vector<OnScreenKeyboard::Key> row;
  for (const char* c = chars; *c; ++c) {
    row.push_back({OnScreenKeyboard::KeyType::kChar, *c, 1});
  }
  return row;
}
}  // namespace

OnScreenKeyboard::OnScreenKeyboard(std::string text, size_t max_length)
    : text_(std::move(text)), max_length_(max_length) {}

const std::vector<std::vector<OnScreenKeyboard::Key>>& OnScreenKeyboard::Layout() {
  static const std::vector<std::vector<Key>> layout = {
      CharRow("1234567890"),
      CharRow("QWERTYUIOP"),
      CharRow("ASDFGHJKL'"),
      CharRow("ZXCVBNM-.!"),
      {{KeyType::kShift, 0, 2},
       {KeyType::kSpace, ' ', 3},
       {KeyType::kBackspace, 0, 2},
       {KeyType::kOk, 0, 3}},
  };
  return layout;
}

int OnScreenKeyboard::KeyColumn(size_t row, size_t key) {
  int column = 0;
  for (size_t i = 0; i < key; ++i) {
    column += Layout()[row][i].span;
  }
  return column;
}

size_t OnScreenKeyboard::KeyAtColumn(size_t row, int column) {
  const auto& keys = Layout()[row];
  int start = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    if (column < start + keys[i].span) {
      return i;
    }
    start += keys[i].span;
  }
  return keys.size() - 1;
}

void OnScreenKeyboard::Move(int dx, int dy) {
  const auto& layout = Layout();
  if (dx) {
    const int count = int(layout[row_].size());
    key_ = size_t(((int(key_) + dx) % count + count) % count);
    column_ = KeyColumn(row_, key_);
  }
  if (dy) {
    const int rows = int(layout.size());
    row_ = size_t(((int(row_) + dy) % rows + rows) % rows);
    key_ = KeyAtColumn(row_, column_);
  }
}

OnScreenKeyboard::Result OnScreenKeyboard::Press() {
  const Key& key = Layout()[row_][key_];
  switch (key.type) {
    case KeyType::kChar:
      TypeChar(upper_ ? key.ch : char(std::tolower(static_cast<unsigned char>(key.ch))));
      break;
    case KeyType::kShift:
      upper_ = !upper_;
      break;
    case KeyType::kSpace:
      Space();
      break;
    case KeyType::kBackspace:
      Backspace();
      break;
    case KeyType::kOk:
      return Result::kDone;
  }
  return Result::kNone;
}

OnScreenKeyboard::Result OnScreenKeyboard::Back() {
  if (text_.empty()) {
    return Result::kCancel;
  }
  Backspace();
  return Result::kNone;
}

void OnScreenKeyboard::Space() { TypeChar(' '); }

bool OnScreenKeyboard::TypeChar(char c) {
  const unsigned char u = static_cast<unsigned char>(c);
  const bool allowed = (u < 0x80 && std::isalnum(u)) || (c && std::strchr(" '-.!", c));
  if (!allowed || text_.size() >= max_length_) {
    return false;
  }
  text_ += c;
  return true;
}

void OnScreenKeyboard::Backspace() {
  if (!text_.empty()) {
    text_.pop_back();
  }
}

}  // namespace rex::ui
