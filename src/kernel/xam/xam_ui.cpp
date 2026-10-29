/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <rex/ui/on_screen_keyboard.h>
#include <rex/logging.h>
#include <rex/runtime.h>
#include <rex/string.h>
#include <rex/system/flags.h>
#include <rex/system/kernel_state.h>

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

REXCVAR_DEFINE_BOOL(headless, false, "Kernel",
                    "Don't display any UI, using defaults for prompts as needed");
REXCVAR_DEFINE_STRING(headless_keyboard_text, "", "Kernel",
                      "With headless, text returned for keyboard prompts that have no default");
#include <rex/kernel/xam/private.h>
#include <rex/hook.h>
#include <rex/types.h>
#include <rex/system/xtypes.h>
#include <rex/thread.h>
#include <rex/ui/imgui_dialog.h>
#include <rex/ui/imgui_drawer.h>
#include <rex/ui/window.h>
#include <rex/ui/windowed_app_context.h>

namespace rex {
namespace kernel {
namespace xam {
using namespace rex::system;

// TODO(gibbed): This is all one giant WIP that seems to work better than the
// previous immediate synchronous completion of dialogs.
//
// The deferred execution of dialog handling is done in such a way that there is
// a pre-, peri- (completion), and post- callback steps.
//
// pre();
// result = completion();
// CompleteOverlapped(result);
// post();
//
// There are games that are batshit insane enough to wait for the X_OVERLAPPED
// to be completed (ie not X_ERROR_PENDING) before creating a listener to
// receive a notification, which is why we have distinct pre- and post- steps.
//
// We deliberately delay the XN_SYS_UI = false notification to give games time
// to create a listener (if they're insane enough do this).

extern std::atomic<int> xam_dialogs_shown_;

class XamDialog : public rex::ui::ImGuiDialog {
 public:
  void set_close_callback(std::function<void()> close_callback) {
    close_callback_ = close_callback;
  }

 protected:
  XamDialog(rex::ui::ImGuiDrawer* imgui_drawer) : rex::ui::ImGuiDialog(imgui_drawer) {}

  void OnClose() override {
    if (close_callback_) {
      close_callback_();
    }
  }

 private:
  std::function<void()> close_callback_ = nullptr;
};

template <typename T>
X_RESULT xeXamDispatchDialog(T* dialog, std::function<X_RESULT(T*)> close_callback,
                             uint32_t overlapped) {
  auto pre = []() {
    // Broadcast XN_SYS_UI = true
    REX_KERNEL_STATE()->BroadcastNotification(0x9, true);
  };
  auto run = [dialog, close_callback]() -> X_RESULT {
    X_RESULT result;
    dialog->set_close_callback(
        [&dialog, &result, &close_callback]() { result = close_callback(dialog); });
    rex::thread::Fence fence;
    rex::ui::WindowedAppContext* app_context = REX_KERNEL_STATE()->emulator()->app_context();
    if (app_context &&
        app_context->CallInUIThreadSynchronous([&dialog, &fence]() { dialog->Then(&fence); })) {
      ++xam_dialogs_shown_;
      fence.Wait();
      --xam_dialogs_shown_;
    } else {
      delete dialog;
    }
    // dialog should be deleted at this point!
    return result;
  };
  auto post = []() {
    rex::thread::Sleep(std::chrono::milliseconds(100));
    // Broadcast XN_SYS_UI = false
    REX_KERNEL_STATE()->BroadcastNotification(0x9, false);
  };
  if (!overlapped) {
    pre();
    auto result = run();
    post();
    return result;
  } else {
    REX_KERNEL_STATE()->CompleteOverlappedDeferred(run, overlapped, pre, post);
    return X_ERROR_IO_PENDING;
  }
}

template <typename T>
X_RESULT xeXamDispatchDialogEx(T* dialog,
                               std::function<X_RESULT(T*, uint32_t&, uint32_t&)> close_callback,
                               uint32_t overlapped) {
  auto pre = []() {
    // Broadcast XN_SYS_UI = true
    REX_KERNEL_STATE()->BroadcastNotification(0x9, true);
  };
  auto run = [dialog, close_callback](uint32_t& extended_error, uint32_t& length) -> X_RESULT {
    rex::ui::WindowedAppContext* app_context = REX_KERNEL_STATE()->emulator()->app_context();
    X_RESULT result;
    dialog->set_close_callback([&dialog, &result, &extended_error, &length, &close_callback]() {
      result = close_callback(dialog, extended_error, length);
    });
    rex::thread::Fence fence;
    if (app_context &&
        app_context->CallInUIThreadSynchronous([&dialog, &fence]() { dialog->Then(&fence); })) {
      ++xam_dialogs_shown_;
      fence.Wait();
      --xam_dialogs_shown_;
    } else {
      delete dialog;
    }
    // dialog should be deleted at this point!
    return result;
  };
  auto post = []() {
    rex::thread::Sleep(std::chrono::milliseconds(100));
    // Broadcast XN_SYS_UI = false
    REX_KERNEL_STATE()->BroadcastNotification(0x9, false);
  };
  if (!overlapped) {
    pre();
    uint32_t extended_error, length;
    auto result = run(extended_error, length);
    post();
    // TODO(gibbed): do something with extended_error/length?
    return result;
  } else {
    REX_KERNEL_STATE()->CompleteOverlappedDeferredEx(run, overlapped, pre, post);
    return X_ERROR_IO_PENDING;
  }
}

X_RESULT xeXamDispatchHeadless(std::function<X_RESULT()> run_callback, uint32_t overlapped) {
  auto pre = []() {
    REXKRNL_DEBUG("xeXamDispatchHeadless: Broadcasting XN_SYS_UI = true");
    // Broadcast XN_SYS_UI = true
    REX_KERNEL_STATE()->BroadcastNotification(0x9, true);
  };
  auto post = []() {
    rex::thread::Sleep(std::chrono::milliseconds(100));
    REXKRNL_DEBUG("xeXamDispatchHeadless: Broadcasting XN_SYS_UI = false");
    // Broadcast XN_SYS_UI = false
    REX_KERNEL_STATE()->BroadcastNotification(0x9, false);
  };
  if (!overlapped) {
    pre();
    auto result = run_callback();
    post();
    return result;
  } else {
    REX_KERNEL_STATE()->CompleteOverlappedDeferred(run_callback, overlapped, pre, post);
    return X_ERROR_IO_PENDING;
  }
}

X_RESULT xeXamDispatchHeadlessEx(std::function<X_RESULT(uint32_t&, uint32_t&)> run_callback,
                                 uint32_t overlapped) {
  auto pre = []() {
    // Broadcast XN_SYS_UI = true
    REX_KERNEL_STATE()->BroadcastNotification(0x9, true);
  };
  auto post = []() {
    rex::thread::Sleep(std::chrono::milliseconds(100));
    // Broadcast XN_SYS_UI = false
    REX_KERNEL_STATE()->BroadcastNotification(0x9, false);
  };
  if (!overlapped) {
    pre();
    uint32_t extended_error, length;
    auto result = run_callback(extended_error, length);
    post();
    // TODO(gibbed): do something with extended_error/length?
    return result;
  } else {
    REX_KERNEL_STATE()->CompleteOverlappedDeferredEx(run_callback, overlapped, pre, post);
    return X_ERROR_IO_PENDING;
  }
}

u32 XamIsUIActive_entry() {
  return xeXamIsUIActive();
}

class MessageBoxDialog : public XamDialog {
 public:
  MessageBoxDialog(rex::ui::ImGuiDrawer* imgui_drawer, std::string title, std::string description,
                   std::vector<std::string> buttons, uint32_t default_button)
      : XamDialog(imgui_drawer),
        title_(title),
        description_(description),
        buttons_(std::move(buttons)),
        default_button_(default_button),
        chosen_button_(default_button) {
    if (!title_.size()) {
      title_ = "Message Box";
    }
  }

  uint32_t chosen_button() const { return chosen_button_; }

  void OnDraw(ImGuiIO& io) override {
    bool first_draw = false;
    if (!has_opened_) {
      ImGui::OpenPopup(title_.c_str());
      has_opened_ = true;
      first_draw = true;
    }
    if (ImGui::BeginPopupModal(title_.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
      if (description_.size()) {
        ImGui::Text("%s", description_.c_str());
      }
      if (first_draw) {
        ImGui::SetKeyboardFocusHere();
      }
      for (size_t i = 0; i < buttons_.size(); ++i) {
        if (ImGui::Button(buttons_[i].c_str())) {
          chosen_button_ = static_cast<uint32_t>(i);
          ImGui::CloseCurrentPopup();
          Close();
        }
        ImGui::SameLine();
      }
      ImGui::Spacing();
      ImGui::Spacing();
      ImGui::EndPopup();
    } else {
      Close();
    }
  }

 private:
  bool has_opened_ = false;
  std::string title_;
  std::string description_;
  std::vector<std::string> buttons_;
  uint32_t default_button_ = 0;
  uint32_t chosen_button_ = 0;
};

// https://www.se7ensins.com/forums/threads/working-xshowmessageboxui.844116/
u32 XamShowMessageBoxUI_entry(u32 user_index, mapped_wstring title_ptr, mapped_wstring text_ptr,
                              u32 button_count, mapped_u32 button_ptrs, u32 active_button,
                              u32 flags, mapped_u32 result_ptr, mapped_void overlapped) {
  REXKRNL_DEBUG(
      "XamShowMessageBoxUI({:08X}, {:08X}, {:08X}, {:08X}, {:08X}, {:08X}, {:08X}, {:08X}, {:08X})",
      uint32_t(user_index), title_ptr.guest_address(), text_ptr.guest_address(),
      uint32_t(button_count), button_ptrs.guest_address(), uint32_t(active_button), uint32_t(flags),
      result_ptr.guest_address(), overlapped.guest_address());
  std::string title;
  if (title_ptr) {
    title = rex::string::to_utf8(rex::memory::load_and_swap<std::u16string>(
        REX_KERNEL_MEMORY()->TranslateVirtual(title_ptr.guest_address())));
  } else {
    title = "";  // TODO(gibbed): default title based on flags?
  }
  std::string text_str = text_ptr
                             ? rex::string::to_utf8(rex::memory::load_and_swap<std::u16string>(
                                   REX_KERNEL_MEMORY()->TranslateVirtual(text_ptr.guest_address())))
                             : "";

  std::vector<std::string> buttons;
  for (uint32_t i = 0; i < button_count; ++i) {
    uint32_t button_ptr = button_ptrs[i];
    auto button = rex::memory::load_and_swap<std::u16string>(
        REX_KERNEL_MEMORY()->TranslateVirtual(button_ptr));
    buttons.push_back(rex::string::to_utf8(button));
  }

  X_RESULT result;
  if (REXCVAR_GET(headless)) {
    // Auto-pick the focused button.
    auto run = [result_ptr, active_button]() -> X_RESULT {
      *result_ptr = static_cast<uint32_t>(active_button);
      return X_ERROR_SUCCESS;
    };
    result = xeXamDispatchHeadless(run, overlapped.guest_address());
  } else {
    // TODO(benvanik): setup icon states.
    switch (flags & 0xF) {
      case 0:
        // config.pszMainIcon = nullptr;
        break;
      case 1:
        // config.pszMainIcon = TD_ERROR_ICON;
        break;
      case 2:
        // config.pszMainIcon = TD_WARNING_ICON;
        break;
      case 3:
        // config.pszMainIcon = TD_INFORMATION_ICON;
        break;
    }
    auto close = [result_ptr](MessageBoxDialog* dialog) -> X_RESULT {
      *result_ptr = dialog->chosen_button();
      return X_ERROR_SUCCESS;
    };
    const Runtime* emulator = REX_KERNEL_STATE()->emulator();
    ui::ImGuiDrawer* imgui_drawer = emulator->imgui_drawer();
    if (imgui_drawer) {
      result = xeXamDispatchDialog<MessageBoxDialog>(
          new MessageBoxDialog(imgui_drawer, title, text_str, buttons, active_button), close,
          overlapped.guest_address());
    } else {
      // Fallback to headless if no drawer available
      auto run = [result_ptr, active_button]() -> X_RESULT {
        *result_ptr = static_cast<uint32_t>(active_button);
        return X_ERROR_SUCCESS;
      };
      result = xeXamDispatchHeadless(run, overlapped.guest_address());
    }
  }
  return result;
}

#ifdef _WIN32
// Buttons held on any XInput controller. Game input is paused while a system
// dialog is open, so dialogs read the pad directly.
uint16_t PadButtonsForDialog() {
  struct XInputState {
    uint32_t packet;
    uint16_t buttons;
    uint8_t triggers[2];
    int16_t thumbs[4];
  };
  using GetStateFn = uint32_t(__stdcall*)(uint32_t, XInputState*);
  static GetStateFn get_state = [] {
    for (const char* name : {"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"}) {
      if (HMODULE module = LoadLibraryA(name)) {
        return reinterpret_cast<GetStateFn>(GetProcAddress(module, "XInputGetState"));
      }
    }
    return GetStateFn(nullptr);
  }();
  uint16_t buttons = 0;
  for (uint32_t user = 0; get_state && user < 4; ++user) {
    XInputState state{};
    if (get_state(user, &state) == 0) {
      buttons |= state.buttons;
    }
  }
  return buttons;
}
#else
uint16_t PadButtonsForDialog() { return 0; }
#endif

// The rider-name prompt: a "Night session" panel over the game with an
// on-screen QWERTY keyboard for controllers (rex::ui::OnScreenKeyboard); a
// physical keyboard types straight into the name. Fonts are the game's bundled
// Exo 2 ("ReAmped Display" / "ReAmped UI" / "ReAmped UI Bold", added by the
// app), falling back to the default font.
class KeyboardInputDialog : public XamDialog {
 public:
  KeyboardInputDialog(rex::ui::ImGuiDrawer* imgui_drawer, std::string title,
                      std::string description, std::string default_text, size_t max_length)
      : XamDialog(imgui_drawer),
        title_(title),
        description_(description),
        default_text_(default_text),
        max_length_(max_length),
        // max_length includes the terminator (and a spare): characters that fit.
        keyboard_(std::string(), max_length > 2 ? max_length - 2 : 0) {
    if (!title_.size()) {
      if (!description_.size()) {
        title_ = "Keyboard Input";
      } else {
        title_ = description_;
        description_ = "";
      }
    }
    if (default_text_.empty()) {
      default_text_ = REXCVAR_GET(headless_keyboard_text);
    }
    for (char c : default_text_) {
      keyboard_.TypeChar(c);
    }
    text_ = keyboard_.text();
  }

  const std::string& text() const { return text_; }
  bool cancelled() const { return cancelled_; }

  void OnDraw(ImGuiIO& io) override {
    if (!has_opened_) {
      ImGui::OpenPopup(title_.c_str());
      has_opened_ = true;
    }
    const float scale = io.DisplaySize.y / 1080.0f;
    const float width = std::clamp(io.DisplaySize.x * 0.36f, 520.0f * scale, 900.0f * scale);
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(width, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, Color(kNavy));
    ImGui::PushStyleColor(ImGuiCol_Border, Color(kBorder));
    ImGui::PushStyleColor(ImGuiCol_ModalWindowDimBg, IM_COL32(4, 8, 14, 150));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 12.0f * scale);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    if (ImGui::BeginPopupModal(title_.c_str(), nullptr,
                               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                                   ImGuiWindowFlags_NoScrollbar)) {
      HandleInput(io);
      Draw(scale, width);
      if (closing_ && !(PadButtons() & (kPadA | kPadB | kPadStart))) {
        // Closed once the confirming button is released: game input resumes
        // when the dialog closes, and a still-held A would reopen the prompt.
        text_ = keyboard_.text();
        ImGui::CloseCurrentPopup();
        Close();
      }
      ImGui::EndPopup();
    } else {
      Close();
    }
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(3);
  }

 private:
  static constexpr uint32_t kNavy = 0x0f1724, kHeader = 0x1c2e48, kField = 0x18253a,
                            kBorder = 0x263852, kCyan = 0x5fe1f0, kOrange = 0xff7a29,
                            kText = 0xe6eef6, kMuted = 0x7f97b3, kKeyText = 0xc6d4e3;
  static constexpr uint16_t kPadUp = 0x0001, kPadDown = 0x0002, kPadLeft = 0x0004,
                            kPadRight = 0x0008, kPadStart = 0x0010, kPadA = 0x1000,
                            kPadB = 0x2000, kPadX = 0x4000;

  static ImU32 Color(uint32_t rgb, uint8_t alpha = 255) {
    return IM_COL32((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF, alpha);
  }
  static ImFont* FindFont(const char* name) {
    for (ImFont* font : ImGui::GetIO().Fonts->Fonts) {
      if (std::strcmp(font->GetDebugName(), name) == 0) {
        return font;
      }
    }
    return nullptr;
  }
  // D-pad buttons, with the left stick (past half way) as the D-pad.
  static uint16_t PadButtons() { return PadButtonsForDialog(); }

  static int TypedCharFilter(ImGuiInputTextCallbackData* data) {
    auto* self = static_cast<KeyboardInputDialog*>(data->UserData);
    if (!self->closing_ && data->EventChar < 0x80) {
      self->keyboard_.TypeChar(char(data->EventChar));
    }
    return 1;  // Discard: the model holds the text.
  }

  void Finish(bool cancelled) {
    cancelled_ = cancelled;
    closing_ = true;
  }

  void HandleInput(ImGuiIO& io) {
    if (closing_) {
      return;
    }
    // Physical keyboard (characters arrive through the hidden text field in
    // Draw, which keeps SDL text input on).
    (void)io;
    if (ImGui::IsKeyPressed(ImGuiKey_Backspace)) {
      keyboard_.Backspace();
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) ||
        ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)) {
      Finish(false);
      return;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
      Finish(true);
      return;
    }
    // Controller: nothing counts until every button was released once (the
    // press that opened the prompt doesn't type or confirm).
    const uint16_t pad = PadButtons() | StickAsDpad();
    const uint16_t all = kPadUp | kPadDown | kPadLeft | kPadRight | kPadStart | kPadA | kPadB |
                         kPadX;
    if (!pad_armed_) {
      pad_armed_ = (pad & all) == 0;
      last_pad_ = pad;
      return;
    }
    const uint16_t pressed = pad & ~last_pad_;
    const double now = ImGui::GetTime();
    const uint16_t dirs = pad & (kPadUp | kPadDown | kPadLeft | kPadRight);
    bool move = false;
    if (dirs != (last_pad_ & (kPadUp | kPadDown | kPadLeft | kPadRight))) {
      move = dirs != 0;
      next_repeat_ = now + 0.35;
    } else if (dirs && now >= next_repeat_) {
      move = true;
      next_repeat_ = now + 0.09;
    }
    if (move) {
      keyboard_.Move((dirs & kPadRight ? 1 : 0) - (dirs & kPadLeft ? 1 : 0),
                     (dirs & kPadDown ? 1 : 0) - (dirs & kPadUp ? 1 : 0));
    }
    last_pad_ = pad;
    if (pressed & kPadA) {
      if (keyboard_.Press() == rex::ui::OnScreenKeyboard::Result::kDone) {
        Finish(false);
      }
    } else if (pressed & kPadB) {
      if (keyboard_.Back() == rex::ui::OnScreenKeyboard::Result::kCancel) {
        Finish(true);
      }
    } else if (pressed & kPadX) {
      keyboard_.Space();
    } else if (pressed & kPadStart) {
      Finish(false);
    }
  }

  static uint16_t StickAsDpad() {
#if REX_PLATFORM_WIN32
    struct XInputState {
      uint32_t packet;
      uint16_t buttons;
      uint8_t triggers[2];
      int16_t thumbs[4];
    };
    using GetStateFn = uint32_t(__stdcall*)(uint32_t, XInputState*);
    static GetStateFn get_state = [] {
      for (const char* name : {"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"}) {
        if (HMODULE module = LoadLibraryA(name)) {
          return reinterpret_cast<GetStateFn>(GetProcAddress(module, "XInputGetState"));
        }
      }
      return GetStateFn(nullptr);
    }();
    uint16_t dirs = 0;
    for (uint32_t user = 0; get_state && user < 4; ++user) {
      XInputState state{};
      if (get_state(user, &state) != 0) continue;
      const int x = state.thumbs[0], y = state.thumbs[1];
      if (x > 16384) dirs |= kPadRight;
      if (x < -16384) dirs |= kPadLeft;
      if (y > 16384) dirs |= kPadUp;
      if (y < -16384) dirs |= kPadDown;
    }
    return dirs;
#else
    return 0;
#endif
  }

  void Draw(float scale, float width) {
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetWindowPos();
    const float pad = 24.0f * scale;
    // Typing on a physical keyboard: SDL sends text only while an ImGui text
    // field is active, so an invisible one keeps focus; its character filter
    // feeds the keyboard model and discards the character from the field.
    ImGui::SetCursorPos(ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.0f);
    ImGui::SetNextItemWidth(1.0f);
    if (!ImGui::IsAnyItemActive()) {
      ImGui::SetKeyboardFocusHere();
    }
    char typing[2] = {};
    ImGui::InputText("##typing", typing, sizeof(typing), ImGuiInputTextFlags_CallbackCharFilter,
                     &TypedCharFilter, this);
    ImGui::PopStyleVar();
    ImGui::SetCursorPos(ImVec2(0.0f, 0.0f));

    ImFont* display = FindFont("ReAmped Display");
    ImFont* ui = FindFont("ReAmped UI");
    ImFont* ui_bold = FindFont("ReAmped UI Bold");
    const float rounding = 12.0f * scale;

    // Header: gradient with a faint, static snowfall speckle.
    const float header_h = 150.0f * scale;
    draw->AddRectFilledMultiColor(origin, ImVec2(origin.x + width, origin.y + header_h),
                                  Color(kHeader), Color(kHeader), Color(kNavy), Color(kNavy));
    for (int i = 0; i < 70; ++i) {
      const float fx = float((i * 7919) % 1000) / 1000.0f;
      const float fy = float((i * 104729) % 1000) / 1000.0f;
      draw->AddCircleFilled(ImVec2(origin.x + fx * width, origin.y + fy * header_h),
                            (i % 3 == 0 ? 1.4f : 1.0f) * scale, Color(0xffffff, 45));
    }
    float y = origin.y + pad;

    // Title (the game's prompt), small and upper case.
    std::string title = title_;
    for (char& c : title) c = char(std::toupper(static_cast<unsigned char>(c)));
    ImGui::PushFont(ui_bold, 20.0f * scale);
    draw->AddText(ImVec2(origin.x + pad, y), Color(kMuted), title.c_str());
    ImGui::PopFont();
    y += 34.0f * scale;

    // The name, large, in a cyan-bordered field with an orange caret.
    const float field_h = 72.0f * scale;
    const ImVec2 f0(origin.x + pad, y), f1(origin.x + width - pad, y + field_h);
    draw->AddRectFilled(f0, f1, Color(kField), 8.0f * scale);
    draw->AddRect(f0, f1, Color(kCyan), 8.0f * scale, 0, 2.0f * scale);
    const float name_size = 50.0f * scale;
    ImGui::PushFont(display, name_size);
    const ImVec2 text_size = ImGui::CalcTextSize(keyboard_.text().c_str());
    const ImVec2 text_pos(f0.x + 16.0f * scale, f0.y + (field_h - name_size) * 0.5f);
    draw->AddText(text_pos, Color(0xffffff), keyboard_.text().c_str());
    ImGui::PopFont();
    if (std::fmod(ImGui::GetTime(), 1.0) < 0.6) {
      const float cx = text_pos.x + text_size.x + 4.0f * scale;
      draw->AddRectFilled(ImVec2(cx, f0.y + 14.0f * scale), ImVec2(cx + 3.0f * scale, f1.y - 14.0f * scale),
                          Color(kOrange));
    }
    y += field_h + 6.0f * scale;
    const std::string count = fmt::format("{} / {}", keyboard_.text().size(),
                                          max_length_ > 2 ? max_length_ - 2 : 0);
    ImGui::PushFont(ui, 18.0f * scale);
    const ImVec2 count_size = ImGui::CalcTextSize(count.c_str());
    draw->AddText(ImVec2(f1.x - count_size.x, y), Color(kMuted), count.c_str());
    ImGui::PopFont();
    y += 30.0f * scale;

    // Keyboard grid (10 columns).
    const auto& layout = rex::ui::OnScreenKeyboard::Layout();
    const float gap = 6.0f * scale;
    const float key_w = (width - 2.0f * pad - 9.0f * gap) / 10.0f;
    const float key_h = 46.0f * scale;
    ImGui::PushFont(ui_bold, 22.0f * scale);
    for (size_t r = 0; r < layout.size(); ++r) {
      float x = origin.x + pad;
      for (size_t k = 0; k < layout[r].size(); ++k) {
        const auto& key = layout[r][k];
        const float w = key_w * float(key.span) + gap * float(key.span - 1);
        const ImVec2 k0(x, y), k1(x + w, y + key_h);
        const bool highlighted = r == keyboard_.row() && k == keyboard_.key();
        const bool ok = key.type == rex::ui::OnScreenKeyboard::KeyType::kOk;
        uint32_t fill = ok ? kOrange : kField, label_color = ok ? 0xffffff : kKeyText;
        if (highlighted) {
          fill = kCyan;
          label_color = kNavy;
        }
        draw->AddRectFilled(k0, k1, Color(fill), 6.0f * scale);
        if (!highlighted) {
          draw->AddRect(k0, k1, Color(ok ? kOrange : kBorder), 6.0f * scale);
        }
        std::string label;
        switch (key.type) {
          case rex::ui::OnScreenKeyboard::KeyType::kChar:
            label.assign(1, keyboard_.upper()
                                ? key.ch
                                : char(std::tolower(static_cast<unsigned char>(key.ch))));
            break;
          case rex::ui::OnScreenKeyboard::KeyType::kShift:
            label = keyboard_.upper() ? "abc" : "ABC";
            break;
          case rex::ui::OnScreenKeyboard::KeyType::kSpace:
            label = "SPACE";
            break;
          case rex::ui::OnScreenKeyboard::KeyType::kBackspace:
            label = "DEL";
            break;
          case rex::ui::OnScreenKeyboard::KeyType::kOk:
            label = "OK";
            break;
        }
        const ImVec2 ls = ImGui::CalcTextSize(label.c_str());
        draw->AddText(ImVec2(k0.x + (w - ls.x) * 0.5f, k0.y + (key_h - ls.y) * 0.5f),
                      Color(label_color), label.c_str());
        // Mouse: click a key.
        if (!closing_ && ImGui::IsMouseHoveringRect(k0, k1) && ImGui::IsMouseClicked(0)) {
          keyboard_.Move(0, 0);
          while (keyboard_.row() != r) keyboard_.Move(0, 1);
          while (keyboard_.key() != k) keyboard_.Move(1, 0);
          if (keyboard_.Press() == rex::ui::OnScreenKeyboard::Result::kDone) {
            Finish(false);
          }
        }
        x += w + gap;
      }
      y += key_h + gap;
    }
    ImGui::PopFont();

    // Controller hints.
    y += 8.0f * scale;
    struct Hint {
      const char* glyph;
      uint32_t color;
      const char* text;
    };
    const Hint hints[] = {{"A", 0x3fae3f, "Type"},
                          {"B", 0xd43b3b, "Delete"},
                          {"X", 0x2f6fd6, "Space"},
                          {"=", 0x6b7685, "Done"}};
    ImGui::PushFont(ui, 18.0f * scale);
    float hints_w = 0.0f;
    for (const Hint& h : hints) {
      hints_w += 26.0f * scale + ImGui::CalcTextSize(h.text).x + 22.0f * scale;
    }
    float hx = origin.x + (width - hints_w) * 0.5f;
    const float radius = 10.0f * scale;
    for (const Hint& h : hints) {
      const ImVec2 c(hx + radius, y + radius);
      draw->AddCircleFilled(c, radius, Color(h.color));
      const ImVec2 gs = ImGui::CalcTextSize(h.glyph);
      draw->AddText(ImVec2(c.x - gs.x * 0.5f, c.y - gs.y * 0.5f), Color(0xffffff), h.glyph);
      draw->AddText(ImVec2(hx + 26.0f * scale, y + radius - ImGui::CalcTextSize(h.text).y * 0.5f),
                    Color(kKeyText), h.text);
      hx += 26.0f * scale + ImGui::CalcTextSize(h.text).x + 22.0f * scale;
    }
    ImGui::PopFont();
    y += 2.0f * radius + pad;
    (void)rounding;
    // Reserve the drawn height in the window.
    ImGui::Dummy(ImVec2(width, y - origin.y));
  }

  bool has_opened_ = false;
  std::string title_;
  std::string description_;
  std::string default_text_;
  size_t max_length_ = 0;
  rex::ui::OnScreenKeyboard keyboard_;
  bool pad_armed_ = false;
  uint16_t last_pad_ = 0;
  double next_repeat_ = 0.0;
  bool closing_ = false;
  std::string text_ = "";
  bool cancelled_ = true;
};

// https://www.se7ensins.com/forums/threads/release-how-to-use-xshowkeyboardui-release.906568/
u32 XamShowKeyboardUI_entry(u32 user_index, u32 flags, mapped_wstring default_text,
                            mapped_wstring title, mapped_wstring description, mapped_wstring buffer,
                            u32 buffer_length, mapped_void overlapped) {
  REXKRNL_DEBUG("XamShowKeyboardUI({:08X}, {:08X}, {:08X}, {:08X}, {:08X}, {:08X}, {:08X}, {:08X})",
                uint32_t(user_index), uint32_t(flags), default_text.guest_address(),
                title.guest_address(), description.guest_address(), buffer.guest_address(),
                uint32_t(buffer_length), overlapped.guest_address());
  if (!buffer) {
    return X_ERROR_INVALID_PARAMETER;
  }

  assert_true(overlapped.guest_address() != 0);

  auto buffer_size = static_cast<size_t>(buffer_length) * 2;

  X_RESULT result;
  if (REXCVAR_GET(headless)) {
    auto run = [default_text, buffer, buffer_length, buffer_size]() -> X_RESULT {
      // Redirect default_text back into the buffer.
      const std::string fallback = REXCVAR_GET(headless_keyboard_text);
      if ((!default_text || default_text.value().empty()) && !fallback.empty()) {
        rex::string::copy_and_swap_truncating(buffer, std::u16string(fallback.begin(), fallback.end()),
                                              buffer_length);
      } else if (!default_text) {
        std::memset(buffer, 0, buffer_size);
      } else {
        rex::string::copy_and_swap_truncating(buffer, default_text.value(), buffer_length);
      }
      return X_ERROR_SUCCESS;
    };
    result = xeXamDispatchHeadless(run, overlapped.guest_address());
  } else {
    auto close = [buffer, buffer_length](KeyboardInputDialog* dialog, uint32_t& extended_error,
                                         uint32_t& length) -> X_RESULT {
      if (dialog->cancelled()) {
        extended_error = X_ERROR_CANCELLED;
        length = 0;
        return X_ERROR_SUCCESS;
      } else {
        // Zero the output buffer.
        auto text = rex::string::to_utf16(dialog->text());
        rex::string::copy_and_swap_truncating(buffer, text, buffer_length);
        extended_error = X_ERROR_SUCCESS;
        length = 0;
        // Keep game input paused a little longer: after a name is entered the
        // game spends ~4.5 s creating and reading the save with its menu still
        // live, and a second A there opened another prompt that was left
        // behind when the game moved on.
        ++xam_dialogs_shown_;
        std::thread([] {
          std::this_thread::sleep_for(std::chrono::seconds(5));
          --xam_dialogs_shown_;
        }).detach();
        return X_ERROR_SUCCESS;
      }
    };
    const Runtime* emulator = REX_KERNEL_STATE()->emulator();
    ui::ImGuiDrawer* imgui_drawer = emulator->imgui_drawer();

    // Read and convert title/description/default_text from guest memory as utf16 to utf8 strings
    std::string title_str = title
                                ? rex::string::to_utf8(rex::memory::load_and_swap<std::u16string>(
                                      REX_KERNEL_MEMORY()->TranslateVirtual(title.guest_address())))
                                : "";
    std::string desc_str =
        description ? rex::string::to_utf8(rex::memory::load_and_swap<std::u16string>(
                          REX_KERNEL_MEMORY()->TranslateVirtual(description.guest_address())))
                    : "";
    std::string def_text_str =
        default_text ? rex::string::to_utf8(rex::memory::load_and_swap<std::u16string>(
                           REX_KERNEL_MEMORY()->TranslateVirtual(default_text.guest_address())))
                     : "";

    if (imgui_drawer) {
      uint32_t buffer_length_safe = buffer_length + 1;  // +1 for null terminator, just in case
      result = xeXamDispatchDialogEx<KeyboardInputDialog>(
          new KeyboardInputDialog(imgui_drawer, title_str, desc_str, def_text_str,
                                  buffer_length_safe),
          close, overlapped.guest_address());
    } else {
      // Fallback to headless
      auto run = [default_text, buffer, buffer_length, buffer_size]() -> X_RESULT {
        if (!default_text) {
          std::memset(buffer, 0, buffer_size);
        } else {
          rex::string::copy_and_swap_truncating(buffer, default_text.value(), buffer_length);
        }
        return X_ERROR_SUCCESS;
      };
      result = xeXamDispatchHeadless(run, overlapped.guest_address());
    }
  }
  return result;
}

u32 XamShowDeviceSelectorUI_entry(u32 user_index, u32 content_type, u32 content_flags,
                                  u64 total_requested, mapped_u32 device_id_ptr,
                                  mapped_void overlapped) {
  REXKRNL_DEBUG("XamShowDeviceSelectorUI({:08X}, {:08X}, {:08X}, {:016X}, {:08X}, {:08X})",
                uint32_t(user_index), uint32_t(content_type), uint32_t(content_flags),
                uint64_t(total_requested), device_id_ptr.guest_address(),
                overlapped.guest_address());
  return xeXamDispatchHeadless(
      [device_id_ptr]() -> X_RESULT {
        // NOTE: 0x00000001 is our dummy device ID from xam_content.cc
        *device_id_ptr = 0x00000001;
        return X_ERROR_SUCCESS;
      },
      overlapped.guest_address());
}

void XamShowDirtyDiscErrorUI_entry(u32 user_index) {
  REXKRNL_ERROR("XamShowDirtyDiscErrorUI called! user_index={}", uint32_t(user_index));
  REXKRNL_ERROR("This indicates a disc/file read error - check that all game files exist");

  const Runtime* emulator = REX_KERNEL_STATE()->emulator();
  ui::ImGuiDrawer* imgui_drawer = emulator->imgui_drawer();
  if (imgui_drawer) {
    xeXamDispatchDialog<MessageBoxDialog>(
        new MessageBoxDialog(imgui_drawer, "Disc Read Error",
                             "There's been an issue reading content from the game disc.\nThis is "
                             "likely caused by bad or unimplemented file IO calls.",
                             {"OK"}, 0),
        [](MessageBoxDialog*) -> X_RESULT { return X_ERROR_SUCCESS; }, 0);
  } else {
    // No UI available - log prominently and pause to let user see the error
    REXKRNL_ERROR("===========================================");
    REXKRNL_ERROR("FATAL: Disc Read Error (no UI to display)");
    REXKRNL_ERROR("Check that all game content files are present");
    REXKRNL_ERROR("Missing files or bad mounts cause this error");
    REXKRNL_ERROR("===========================================");
  }
  // This is death, and should never return.
  // TODO(benvanik): cleaner exit.
  exit(1);
}

u32 XamShowPartyUI_entry(u32 r3, u32 r4) {
  return X_ERROR_FUNCTION_FAILED;
}

u32 XamShowCommunitySessionsUI_entry(u32 r3, u32 r4) {
  return X_ERROR_FUNCTION_FAILED;
}

uint32_t XamShowMessageBoxUIEx_entry() {
  // TODO(tomc): implement properly
  static bool warned = false;
  if (!warned) {
    REXKRNL_WARN("[STUB] XamShowMessageBoxUIEx - not implemented");
    warned = true;
  }
  return 0;
}

}  // namespace xam
}  // namespace kernel
}  // namespace rex

REX_EXPORT(__imp__XamIsUIActive, rex::kernel::xam::XamIsUIActive_entry)
REX_EXPORT(__imp__XamShowMessageBoxUI, rex::kernel::xam::XamShowMessageBoxUI_entry)
REX_EXPORT(__imp__XamShowKeyboardUI, rex::kernel::xam::XamShowKeyboardUI_entry)
REX_EXPORT(__imp__XamShowDeviceSelectorUI, rex::kernel::xam::XamShowDeviceSelectorUI_entry)
REX_EXPORT(__imp__XamShowDirtyDiscErrorUI, rex::kernel::xam::XamShowDirtyDiscErrorUI_entry)
REX_EXPORT(__imp__XamShowPartyUI, rex::kernel::xam::XamShowPartyUI_entry)
REX_EXPORT(__imp__XamShowCommunitySessionsUI, rex::kernel::xam::XamShowCommunitySessionsUI_entry)
REX_EXPORT(__imp__XamShowMessageBoxUIEx, rex::kernel::xam::XamShowMessageBoxUIEx_entry)

REX_EXPORT_STUB(__imp__XamIsGuideDisabled);
REX_EXPORT_STUB(__imp__XamIsMessageBoxActive);
REX_EXPORT_STUB(__imp__XamIsNuiUIActive);
REX_EXPORT_STUB(__imp__XamIsSysUiInvokedByTitle);
REX_EXPORT_STUB(__imp__XamIsSysUiInvokedByXenonButton);
REX_EXPORT_STUB(__imp__XamIsUIThread);
REX_EXPORT_STUB(__imp__XamNavigate);
REX_EXPORT_STUB(__imp__XamNavigateBack);
REX_EXPORT_STUB(__imp__XamOverrideHudOpenType);
REX_EXPORT_STUB(__imp__XamShowAchievementDetailsUI);
REX_EXPORT_STUB(__imp__XamShowAchievementsUI);
REX_EXPORT_STUB(__imp__XamShowAchievementsUIEx);
REX_EXPORT_STUB(__imp__XamShowAndWaitForMessageBoxEx);
REX_EXPORT_STUB(__imp__XamShowAvatarAwardGamesUI);
REX_EXPORT_STUB(__imp__XamShowAvatarAwardsUI);
REX_EXPORT_STUB(__imp__XamShowAvatarMiniCreatorUI);
REX_EXPORT_STUB(__imp__XamShowBadDiscErrorUI);
REX_EXPORT_STUB(__imp__XamShowBeaconsUI);
REX_EXPORT_STUB(__imp__XamShowBrandedKeyboardUI);
REX_EXPORT_STUB(__imp__XamShowChangeGamerTileUI);
REX_EXPORT_STUB(__imp__XamShowComplaintUI);
REX_EXPORT_STUB(__imp__XamShowCreateProfileUI);
REX_EXPORT_STUB(__imp__XamShowCreateProfileUIEx);
REX_EXPORT_STUB(__imp__XamShowCsvTransitionUI);
REX_EXPORT_STUB(__imp__XamShowCustomMessageComposeUI);
REX_EXPORT_STUB(__imp__XamShowCustomPlayerListUI);
REX_EXPORT_STUB(__imp__XamShowDirectAcquireUI);
REX_EXPORT_STUB(__imp__XamShowEditProfileUI);
REX_EXPORT_STUB(__imp__XamShowFirstRunWelcomeUI);
REX_EXPORT_STUB(__imp__XamShowFitnessBodyProfileUI);
REX_EXPORT_STUB(__imp__XamShowFitnessClearUI);
REX_EXPORT_STUB(__imp__XamShowFitnessWarnAboutPrivacyUI);
REX_EXPORT_STUB(__imp__XamShowFitnessWarnAboutTimeUI);
REX_EXPORT_STUB(__imp__XamShowFofUI);
REX_EXPORT_STUB(__imp__XamShowForcedNameChangeUI);
REX_EXPORT_STUB(__imp__XamShowFriendRequestUI);
REX_EXPORT_STUB(__imp__XamShowFriendsUI);
REX_EXPORT_STUB(__imp__XamShowFriendsUIp);
REX_EXPORT_STUB(__imp__XamShowGameInviteUI);
REX_EXPORT_STUB(__imp__XamShowGameVoiceChannelUI);
REX_EXPORT_STUB(__imp__XamShowGamerCardUI);
REX_EXPORT_STUB(__imp__XamShowGamerCardUIForXUID);
REX_EXPORT_STUB(__imp__XamShowGamerCardUIForXUIDp);
REX_EXPORT_STUB(__imp__XamShowGamesUI);
REX_EXPORT_STUB(__imp__XamShowGenericOnlineAppUI);
REX_EXPORT_STUB(__imp__XamShowGoldUpgradeUI);
REX_EXPORT_STUB(__imp__XamShowGraduateUserUI);
REX_EXPORT_STUB(__imp__XamShowGuideUI);
REX_EXPORT_STUB(__imp__XamShowJoinPartyUI);
REX_EXPORT_STUB(__imp__XamShowJoinSessionByIdInProgressUI);
REX_EXPORT_STUB(__imp__XamShowJoinSessionInProgressUI);
REX_EXPORT_STUB(__imp__XamShowKeyboardUIMessenger);
REX_EXPORT_STUB(__imp__XamShowLiveSignupUI);
REX_EXPORT_STUB(__imp__XamShowLiveUpsellUI);
REX_EXPORT_STUB(__imp__XamShowLiveUpsellUIEx);
REX_EXPORT_STUB(__imp__XamShowMarketplaceDownloadItemsUI);
REX_EXPORT_STUB(__imp__XamShowMarketplaceGetOrderReceipts);
REX_EXPORT_STUB(__imp__XamShowMarketplacePurchaseOrderUI);
REX_EXPORT_STUB(__imp__XamShowMarketplacePurchaseOrderUIEx);
REX_EXPORT_STUB(__imp__XamShowMarketplaceUI);
REX_EXPORT_STUB(__imp__XamShowMarketplaceUIEx);
REX_EXPORT_STUB(__imp__XamShowMessageBox);
REX_EXPORT_STUB(__imp__XamShowMessageComposeUI);
REX_EXPORT_STUB(__imp__XamShowMessagesUI);
REX_EXPORT_STUB(__imp__XamShowMessagesUIEx);
REX_EXPORT_STUB(__imp__XamShowMessengerUI);
REX_EXPORT_STUB(__imp__XamShowMultiplayerUpgradeUI);
REX_EXPORT_STUB(__imp__XamShowNetworkStorageSyncUI);
REX_EXPORT_STUB(__imp__XamShowNuiAchievementsUI);
REX_EXPORT_STUB(__imp__XamShowNuiCommunitySessionsUI);
REX_EXPORT_STUB(__imp__XamShowNuiControllerRequiredUI);
REX_EXPORT_STUB(__imp__XamShowNuiDeviceSelectorUI);
REX_EXPORT_STUB(__imp__XamShowNuiDirtyDiscErrorUI);
REX_EXPORT_STUB(__imp__XamShowNuiFriendRequestUI);
REX_EXPORT_STUB(__imp__XamShowNuiFriendsUI);
REX_EXPORT_STUB(__imp__XamShowNuiGameInviteUI);
REX_EXPORT_STUB(__imp__XamShowNuiGamerCardUIForXUID);
REX_EXPORT_STUB(__imp__XamShowNuiGamesUI);
REX_EXPORT_STUB(__imp__XamShowNuiGuideUI);
REX_EXPORT_STUB(__imp__XamShowNuiHardwareRequiredUI);
REX_EXPORT_STUB(__imp__XamShowNuiJoinSessionInProgressUI);
REX_EXPORT_STUB(__imp__XamShowNuiMarketplaceDownloadItemsUI);
REX_EXPORT_STUB(__imp__XamShowNuiMarketplaceUI);
REX_EXPORT_STUB(__imp__XamShowNuiMessageBoxUI);
REX_EXPORT_STUB(__imp__XamShowNuiMessagesUI);
REX_EXPORT_STUB(__imp__XamShowNuiPartyUI);
REX_EXPORT_STUB(__imp__XamShowNuiSigninUI);
REX_EXPORT_STUB(__imp__XamShowNuiVideoRichPresenceUI);
REX_EXPORT_STUB(__imp__XamShowOptionalMediaUpdateRequiredUI);
REX_EXPORT_STUB(__imp__XamShowOptionalMediaUpdateRequiredUIEx);
REX_EXPORT_STUB(__imp__XamShowOptionsUI);
REX_EXPORT_STUB(__imp__XamShowPamUI);
REX_EXPORT_STUB(__imp__XamShowPartyInviteUI);
REX_EXPORT_STUB(__imp__XamShowPartyJoinInProgressUI);
REX_EXPORT_STUB(__imp__XamShowPasscodeVerifyUI);
REX_EXPORT_STUB(__imp__XamShowPasscodeVerifyUIEx);
REX_EXPORT_STUB(__imp__XamShowPaymentOptionsUI);
REX_EXPORT_STUB(__imp__XamShowPersonalizationUI);
REX_EXPORT_STUB(__imp__XamShowPlayerReviewUI);
REX_EXPORT_STUB(__imp__XamShowPlayersUI);
REX_EXPORT_STUB(__imp__XamShowPrivateChatInviteUI);
REX_EXPORT_STUB(__imp__XamShowQuickChatUI);
REX_EXPORT_STUB(__imp__XamShowQuickChatUIp);
REX_EXPORT_STUB(__imp__XamShowQuickLaunchUI);
REX_EXPORT_STUB(__imp__XamShowRecentMessageUI);
REX_EXPORT_STUB(__imp__XamShowRecentMessageUIEx);
REX_EXPORT_STUB(__imp__XamShowReputationUI);
REX_EXPORT_STUB(__imp__XamShowSigninUIEx);
REX_EXPORT_STUB(__imp__XamShowSigninUIp);
REX_EXPORT_STUB(__imp__XamShowSignupCreditCardUI);
REX_EXPORT_STUB(__imp__XamShowSocialPostUI);
REX_EXPORT_STUB(__imp__XamShowStorePickerUI);
REX_EXPORT_STUB(__imp__XamShowTFAUI);
REX_EXPORT_STUB(__imp__XamShowTermsOfUseUI);
REX_EXPORT_STUB(__imp__XamShowUpdaterUI);
REX_EXPORT_STUB(__imp__XamShowVideoChatInviteUI);
REX_EXPORT_STUB(__imp__XamShowVideoRichPresenceUI);
REX_EXPORT_STUB(__imp__XamShowVoiceMailUI);
REX_EXPORT_STUB(__imp__XamShowVoiceSettingsUI);
REX_EXPORT_STUB(__imp__XamShowWhatsOnUI);
REX_EXPORT_STUB(__imp__XamShowWordRegisterUI);
REX_EXPORT_STUB(__imp__XamSysUiDisableAutoClose);
