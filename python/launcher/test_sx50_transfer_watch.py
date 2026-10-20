import unittest
from unittest.mock import patch

import sx50_transfer_watch as watch


class TransferWatchTests(unittest.TestCase):
    def test_source_must_be_running_and_duplicate_destinations_refused(self):
        with self.assertRaisesRegex(RuntimeError, "silver-3 source"):
            watch.inventory([])
        source = {"id": watch.SOURCE_ID, "actual_status": "running"}
        self.assertIsNone(watch.inventory([source]))
        with self.assertRaisesRegex(RuntimeError, "multiple transfer rentals"):
            watch.inventory([source, {"label": watch.LABEL}, {"label": watch.LABEL}])

    def test_strict_price_threads_and_gpu(self):
        offer = {"cpu_cores_effective": 191, "dph_total": 0.999,
                 "num_gpus": 1, "gpu_frac": 1, "reliability2": 0.90,
                 "cpu_ram": 128000, "disk_space": 40, "inet_down": 120}
        watch.verify_offer(offer)
        for key, value in (("cpu_cores_effective", 190), ("dph_total", 1),
                           ("num_gpus", 0), ("reliability2", 0.89)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                watch.verify_offer({**offer, key: value})

    @patch.dict(watch.launch.os.environ, {"R2_BUCKET": "test-bucket"})
    @patch.object(watch.launch, "_aws")
    def test_r2_failure_refuses_rental(self, aws):
        aws.return_value.returncode = 255
        aws.return_value.stderr = b"connection failed"
        with self.assertRaisesRegex(RuntimeError, "cannot verify"):
            watch.work_complete()
        aws.return_value.returncode = 1
        aws.return_value.stderr = b"404 Not Found"
        self.assertFalse(watch.work_complete())

    @patch.object(watch.launch, "vast")
    @patch.object(watch, "instances")
    @patch.object(watch.time, "sleep")
    def test_unknown_create_response_never_retries_rental(self, sleep, instances, vast):
        vast.side_effect = RuntimeError("connection lost")
        instances.return_value = [{"id": watch.SOURCE_ID, "actual_status": "running"}]
        with self.assertRaisesRegex(RuntimeError, "inspect inventory"):
            watch.rent({"id": 123})
        self.assertEqual(vast.call_count, 1)
        self.assertEqual(instances.call_count, 6)
        self.assertEqual(sleep.call_count, 5)

    @patch.object(watch, "work_complete", return_value=True)
    @patch.object(watch, "instances", return_value=[{"id": watch.SOURCE_ID, "actual_status": "running"}])
    @patch.object(watch.launch, "choose")
    def test_completed_run_never_searches_or_rents(self, choose, instances, r2_exists):
        self.assertIsNone(watch.watch(hours=0))
        choose.assert_not_called()

    @patch.object(watch.launch, "vast")
    @patch.object(watch, "instances")
    def test_successful_create_requires_visible_matching_instance(self, instances, vast):
        vast.return_value = {"new_contract": 456}
        instances.return_value = [{"id": watch.SOURCE_ID, "actual_status": "running"},
                                  {"id": 456, "label": watch.LABEL}]
        self.assertEqual(watch.rent({"id": 123})["id"], 456)
        args = vast.call_args.args
        self.assertNotIn("--entrypoint", args)
        self.assertNotIn("--env", args)
        self.assertNotIn("--args", args)


if __name__ == "__main__":
    unittest.main()
