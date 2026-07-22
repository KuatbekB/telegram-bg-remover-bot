import asyncio
import unittest
import tempfile
import threading
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image
from telegram.error import NetworkError

import bot


def png_bytes(pixels, size):
    image = Image.new("RGBA", size)
    image.putdata(pixels)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class ImageValidationTests(unittest.TestCase):
    def test_accepts_valid_png(self):
        source = png_bytes([(1, 2, 3, 255)], (1, 1))

        self.assertEqual(bot.validate_image_bytes(source), (1, 1, "PNG"))

    def test_rejects_non_image_bytes(self):
        with self.assertRaises(bot.InvalidImageError):
            bot.validate_image_bytes(b"not an image")

    def test_rejects_too_many_pixels(self):
        source = png_bytes([(1, 2, 3, 255), (4, 5, 6, 255)], (2, 1))

        with patch.object(bot, "MAX_IMAGE_PIXELS", 1), self.assertRaises(bot.InvalidImageError):
            bot.validate_image_bytes(source)

    def test_rejects_unsupported_image_format(self):
        image = Image.new("RGB", (1, 1))
        output = BytesIO()
        image.save(output, format="GIF")

        with self.assertRaises(bot.InvalidImageError):
            bot.validate_image_bytes(output.getvalue())

    def test_rejects_file_larger_than_download_limit(self):
        source = png_bytes([(1, 2, 3, 255)], (1, 1))

        with patch.object(bot, "MAX_DOWNLOAD_BYTES", len(source) - 1), self.assertRaises(bot.InvalidImageError):
            bot.validate_image_bytes(source)


class PolishOutputTests(unittest.TestCase):
    def test_safe_mode_preserves_opaque_white_object_pixels(self):
        source = png_bytes([(255, 255, 255, 255), (20, 20, 20, 255)], (2, 1))

        result = bot.polish_output_png(
            source,
            remove_white_remnants=False,
            feather_radius=0,
        )

        self.assertEqual(Image.open(BytesIO(result)).convert("RGBA").getpixel((0, 0)), (255, 255, 255, 255))

    def test_optional_white_cleanup_remains_explicit(self):
        source = png_bytes([(250, 250, 250, 255), (20, 20, 20, 255)], (2, 1))

        result = bot.polish_output_png(
            source,
            remove_white_remnants=True,
            feather_radius=0,
        )

        self.assertEqual(Image.open(BytesIO(result)).convert("RGBA").getpixel((0, 0))[3], 0)

    @patch("bot.remove")
    @patch("bot.get_rembg_session")
    def test_retry_uses_alternative_model_without_alpha_matting(self, get_session, remove):
        source = png_bytes([(10, 20, 30, 255)], (1, 1))
        get_session.return_value = object()
        remove.return_value = source

        result = bot.remove_background_retry(b"input")

        get_session.assert_called_once_with(bot.RETRY_REMBG_MODEL)
        self.assertFalse(remove.call_args.kwargs["alpha_matting"])
        self.assertEqual(Image.open(BytesIO(result)).convert("RGBA").getpixel((0, 0)), (10, 20, 30, 255))


class ProcessingLimitTests(unittest.TestCase):
    def make_context(self):
        return SimpleNamespace(
            application=SimpleNamespace(bot_data={}),
            user_data={},
        )

    def test_same_user_cannot_start_two_jobs(self):
        context = self.make_context()

        self.assertIsNone(bot.try_begin_processing(context, user_id=10, enforce_cooldown=False))
        reason = bot.try_begin_processing(context, user_id=10, enforce_cooldown=False)

        self.assertIn("в очереди или обрабатывается", reason)

    def test_global_limit_rejects_extra_job(self):
        context = self.make_context()

        with patch.object(bot, "MAX_CONCURRENT_JOBS", 1), patch.object(bot, "MAX_QUEUE_SIZE", 0):
            self.assertIsNone(bot.try_begin_processing(context, user_id=10, enforce_cooldown=False))
            reason = bot.try_begin_processing(context, user_id=20, enforce_cooldown=False)

        self.assertIn("Очередь обработки заполнена", reason)

    def test_finished_job_releases_capacity(self):
        context = self.make_context()

        with patch.object(bot, "MAX_CONCURRENT_JOBS", 1), patch.object(bot, "MAX_QUEUE_SIZE", 0):
            self.assertIsNone(bot.try_begin_processing(context, user_id=10, enforce_cooldown=False))
            bot.finish_processing(context, user_id=10)
            self.assertIsNone(bot.try_begin_processing(context, user_id=20, enforce_cooldown=False))

    def test_cooldown_rejects_fast_repeat(self):
        context = self.make_context()

        with patch.object(bot, "USER_COOLDOWN_SECONDS", 3), patch("bot.time.monotonic", side_effect=[100, 101]):
            self.assertIsNone(bot.try_begin_processing(context, user_id=10, enforce_cooldown=True))
            bot.finish_processing(context, user_id=10)
            reason = bot.try_begin_processing(context, user_id=10, enforce_cooldown=True)

        self.assertIn("Слишком быстро", reason)

    def test_rate_limit_rejects_excess_requests(self):
        context = self.make_context()

        with (
            patch.object(bot, "USER_RATE_LIMIT_COUNT", 2),
            patch.object(bot, "USER_RATE_LIMIT_WINDOW_SECONDS", 60),
            patch("bot.time.monotonic", side_effect=[100, 101, 102]),
        ):
            self.assertIsNone(bot.try_begin_processing(context, user_id=10, enforce_cooldown=False))
            bot.finish_processing(context, user_id=10)
            self.assertIsNone(bot.try_begin_processing(context, user_id=10, enforce_cooldown=False))
            bot.finish_processing(context, user_id=10)
            reason = bot.try_begin_processing(context, user_id=10, enforce_cooldown=False)

        self.assertIn("лимит запросов", reason)


class PrivacyDataTests(unittest.TestCase):
    def test_processing_job_repr_does_not_expose_identifiers(self):
        job = bot.ProcessingJob(
            user_id=123456,
            file_id="sensitive-file-id",
            message=None,
            bot=None,
            user_data={},
            status_message=None,
        )

        self.assertNotIn("123456", repr(job))
        self.assertNotIn("sensitive-file-id", repr(job))

    def test_prune_removes_expired_retry_and_rate_limit_metadata(self):
        user_data = {
            "retry_processing": {"file_id": "file-id", "expires_at": 100},
            "processing_request_times": [100],
            "last_processing_request_at": 100,
        }

        with (
            patch.object(bot, "USER_RATE_LIMIT_WINDOW_SECONDS", 60),
            patch.object(bot, "USER_COOLDOWN_SECONDS", 3),
        ):
            retry_expired = bot.prune_expired_user_data(user_data, now=200)

        self.assertTrue(retry_expired)
        self.assertEqual(user_data, {})

    def test_prune_keeps_unexpired_retry_metadata(self):
        retry_data = {"file_id": "file-id", "expires_at": 200}
        user_data = {"retry_processing": retry_data}

        self.assertFalse(bot.prune_expired_user_data(user_data, now=100))
        self.assertIs(user_data["retry_processing"], retry_data)

    def test_runtime_rejects_unencrypted_debug_storage(self):
        with patch.object(bot, "SAVE_DEBUG_IMAGES", True), self.assertRaises(SystemExit):
            bot.validate_privacy_configuration()


class AsyncProcessingTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_processes_job_and_releases_capacity(self):
        queue = asyncio.Queue()
        application = SimpleNamespace(bot_data={}, drop_user_data=Mock())
        state = bot.new_processing_state(queue)
        state["active_users"].add(10)
        application.bot_data["processing_state"] = state
        job = SimpleNamespace(user_id=10, user_data={"delete_after_processing": True})
        queue.put_nowait(job)

        with patch("bot.process_image_job", new=AsyncMock()) as process_job:
            worker = asyncio.create_task(bot.processing_worker(application, 0))
            await asyncio.wait_for(queue.join(), timeout=1)
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

        process_job.assert_awaited_once_with(job)
        self.assertEqual(state["active_jobs"], 0)
        self.assertNotIn(10, state["active_users"])
        self.assertEqual(job.user_data, {})
        application.drop_user_data.assert_called_once_with(10)

    async def test_workers_cap_parallel_image_processing(self):
        queue = asyncio.Queue()
        application = SimpleNamespace(bot_data={})
        state = bot.new_processing_state(queue)
        state["active_users"].update({1, 2, 3})
        application.bot_data["processing_state"] = state
        for user_id in (1, 2, 3):
            queue.put_nowait(SimpleNamespace(user_id=user_id))

        running = 0
        peak_running = 0

        async def track_processing(_job):
            nonlocal running, peak_running
            running += 1
            peak_running = max(peak_running, running)
            await asyncio.sleep(0.02)
            running -= 1

        with patch("bot.process_image_job", new=track_processing):
            workers = [
                asyncio.create_task(bot.processing_worker(application, index))
                for index in range(2)
            ]
            await asyncio.wait_for(queue.join(), timeout=1)
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        self.assertEqual(peak_running, 2)
        self.assertEqual(state["active_jobs"], 0)
        self.assertEqual(state["active_users"], set())

    async def test_successful_job_records_retry_metadata(self):
        source = png_bytes([(1, 2, 3, 255)], (1, 1))
        output = png_bytes([(1, 2, 3, 0)], (1, 1))
        status_message = SimpleNamespace(edit_text=AsyncMock(), delete=AsyncMock())
        message = SimpleNamespace(chat_id=100)
        telegram_bot = SimpleNamespace(send_chat_action=AsyncMock())
        user_data = {}
        job = bot.ProcessingJob(
            user_id=10,
            file_id="file-id",
            message=message,
            bot=telegram_bot,
            user_data=user_data,
            status_message=status_message,
        )

        with (
            patch("bot.download_job_input", new=AsyncMock(return_value=(source, ".png"))),
            patch("bot.validate_image_bytes"),
            patch("bot.remove_background", return_value=output),
            patch("bot.progress_animation", new=AsyncMock()),
            patch(
                "bot.send_result_document",
                new=AsyncMock(return_value=SimpleNamespace(message_id=123)),
            ),
        ):
            await bot.process_image_job(job)

        self.assertEqual(user_data["retry_processing"]["result_message_id"], 123)
        self.assertEqual(user_data["retry_processing"]["file_id"], "file-id")
        self.assertGreater(user_data["retry_processing"]["expires_at"], time.monotonic())
        status_message.delete.assert_awaited_once()

    async def test_privacy_command_describes_processing_and_deletion(self):
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(effective_message=message)

        with patch.object(bot, "PRIVACY_CONTACT", "<operator>"):
            await bot.privacy_command(update, SimpleNamespace())

        text = message.reply_text.await_args.args[0]
        self.assertIn("/delete_me", text)
        self.assertIn("&lt;operator&gt;", text)
        self.assertIn("не передаётся сторонним AI-сервисам", text)

    async def test_delete_my_data_drops_inactive_user_state(self):
        application = SimpleNamespace(bot_data={}, drop_user_data=Mock())
        application.bot_data["processing_state"] = bot.new_processing_state(asyncio.Queue())
        context = SimpleNamespace(application=application, user_data={"retry_processing": {"file_id": "secret"}})
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(effective_user=SimpleNamespace(id=10), effective_message=message)

        await bot.delete_my_data(update, context)

        self.assertEqual(context.user_data, {})
        application.drop_user_data.assert_called_once_with(10)

    async def test_delete_my_data_defers_final_drop_for_active_job(self):
        application = SimpleNamespace(bot_data={}, drop_user_data=Mock())
        state = bot.new_processing_state(asyncio.Queue())
        state["active_users"].add(10)
        application.bot_data["processing_state"] = state
        context = SimpleNamespace(application=application, user_data={"retry_processing": {"file_id": "secret"}})
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(effective_user=SimpleNamespace(id=10), effective_message=message)

        await bot.delete_my_data(update, context)

        self.assertEqual(context.user_data, {"delete_after_processing": True})
        application.drop_user_data.assert_not_called()

    async def test_privacy_cleanup_drops_empty_inactive_user_state(self):
        application = SimpleNamespace(
            bot_data={"processing_state": bot.new_processing_state(asyncio.Queue())},
            user_data={10: {"retry_processing": {"file_id": "secret", "expires_at": 0}}},
            drop_user_data=Mock(),
        )

        with patch.object(bot, "PRIVACY_CLEANUP_INTERVAL_SECONDS", 0.01):
            cleanup_task = asyncio.create_task(bot.privacy_cleanup_loop(application))
            await asyncio.sleep(0.03)
            cleanup_task.cancel()
            await asyncio.gather(cleanup_task, return_exceptions=True)

        application.drop_user_data.assert_called_with(10)

    async def test_progress_failure_does_not_fail_job(self):
        status_message = SimpleNamespace(
            edit_text=AsyncMock(side_effect=NetworkError("temporary failure"))
        )

        with patch.object(bot, "TELEGRAM_MAX_RETRIES", 0):
            await bot.safe_edit_progress(status_message, "Test", 50, "Still working")

    async def test_timeout_keeps_worker_slot_until_native_processing_finishes(self):
        source = png_bytes([(1, 2, 3, 255)], (1, 1))
        status_message = SimpleNamespace(edit_text=AsyncMock(), delete=AsyncMock())
        finished = threading.Event()

        def slow_processor(_input_bytes):
            time.sleep(0.05)
            finished.set()
            return source

        job = bot.ProcessingJob(
            user_id=10,
            file_id="file-id",
            message=SimpleNamespace(chat_id=100),
            bot=SimpleNamespace(send_chat_action=AsyncMock()),
            user_data={},
            status_message=status_message,
        )

        with (
            patch.object(bot, "PROCESSING_TIMEOUT_SECONDS", 0.01),
            patch("bot.download_job_input", new=AsyncMock(return_value=(source, ".png"))),
            patch("bot.validate_image_bytes"),
            patch("bot.remove_background", side_effect=slow_processor),
            patch("bot.progress_animation", new=AsyncMock()),
            patch("bot.send_result_document", new=AsyncMock()) as send_result,
        ):
            await bot.process_image_job(job)

        self.assertTrue(finished.is_set())
        send_result.assert_not_awaited()
        self.assertIn("Превышено время обработки", status_message.edit_text.await_args.args[0])

    async def test_oversized_result_is_not_sent(self):
        source = png_bytes([(1, 2, 3, 255)], (1, 1))
        status_message = SimpleNamespace(edit_text=AsyncMock(), delete=AsyncMock())
        job = bot.ProcessingJob(
            user_id=10,
            file_id="file-id",
            message=SimpleNamespace(chat_id=100),
            bot=SimpleNamespace(send_chat_action=AsyncMock()),
            user_data={},
            status_message=status_message,
        )

        with (
            patch.object(bot, "MAX_OUTPUT_BYTES", 1),
            patch("bot.download_job_input", new=AsyncMock(return_value=(source, ".png"))),
            patch("bot.validate_image_bytes"),
            patch("bot.remove_background", return_value=b"too large"),
            patch("bot.progress_animation", new=AsyncMock()),
            patch("bot.send_result_document", new=AsyncMock()) as send_result,
        ):
            await bot.process_image_job(job)

        send_result.assert_not_awaited()
        self.assertIn("Результат слишком большой", status_message.edit_text.await_args.args[0])

    async def test_setup_creates_bounded_queue_workers_and_health_files(self):
        application = SimpleNamespace(
            bot_data={},
            bot=SimpleNamespace(set_my_commands=AsyncMock()),
            user_data={},
            drop_user_data=Mock(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            ready_file = Path(temp_dir) / "ready"
            heartbeat_file = Path(temp_dir) / "heartbeat"
            with (
                patch.object(bot, "MAX_CONCURRENT_JOBS", 2),
                patch.object(bot, "MAX_QUEUE_SIZE", 3),
                patch.object(bot, "PRELOAD_MODELS", False),
                patch.object(bot, "PRELOAD_PRIMARY_MODEL", False),
                patch.object(bot, "READY_FILE", ready_file),
                patch.object(bot, "HEARTBEAT_FILE", heartbeat_file),
            ):
                try:
                    await bot.setup_commands(application)
                    await asyncio.sleep(0)
                    state = application.bot_data["processing_state"]
                    self.assertEqual(state["queue"].maxsize, 5)
                    self.assertEqual(len(state["worker_tasks"]), 2)
                    self.assertIsNotNone(state["privacy_cleanup_task"])
                    commands = application.bot.set_my_commands.await_args.args[0]
                    self.assertEqual(
                        {command.command for command in commands},
                        {"start", "help", "privacy", "delete_me"},
                    )
                    self.assertTrue(ready_file.is_file())
                    self.assertTrue(heartbeat_file.is_file())
                finally:
                    await bot.shutdown_app(application)

                self.assertFalse(ready_file.exists())
                self.assertFalse(heartbeat_file.exists())


class StartupTests(unittest.TestCase):
    @patch("bot.get_rembg_session")
    def test_preload_initializes_both_configured_models(self, get_session):
        with patch.object(bot, "REMBG_MODEL", "primary"), patch.object(bot, "RETRY_REMBG_MODEL", "retry"):
            bot.preload_models()

        self.assertEqual([call.args[0] for call in get_session.call_args_list], ["primary", "retry"])

    @patch("bot.get_rembg_session")
    def test_preload_does_not_load_same_model_twice(self, get_session):
        with patch.object(bot, "REMBG_MODEL", "same"), patch.object(bot, "RETRY_REMBG_MODEL", "same"):
            bot.preload_models()

        get_session.assert_called_once_with("same")

    def test_instance_lock_blocks_duplicate_and_can_be_reacquired(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            bot, "INSTANCE_LOCK_FILE", Path(temp_dir) / "bot.lock"
        ):
            try:
                bot.acquire_instance_lock()
                with self.assertRaises(SystemExit):
                    bot.acquire_instance_lock()
            finally:
                bot.release_instance_lock()

            try:
                bot.acquire_instance_lock()
                self.assertTrue(bot.INSTANCE_LOCK_FILE.exists())
            finally:
                bot.release_instance_lock()


if __name__ == "__main__":
    unittest.main()
