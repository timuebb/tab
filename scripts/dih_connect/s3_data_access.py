from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import boto3
import polars as pl
from botocore.config import Config
from botocore.exceptions import ClientError
from deltalake import DeltaTable

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_LOCAL_DATASET_DIR = PROJECT_ROOT / "dataset"

TIMEZONE = ZoneInfo(os.environ.get("APP_TIMEZONE", "Europe/Berlin"))


def _convert_datetimes_to_timezone(lf: pl.LazyFrame, timezone: ZoneInfo) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.col(name).dt.convert_time_zone(str(timezone))
            for name, dtype in lf.collect_schema().items()
            if dtype == pl.Datetime
        ]
    )


def _should_skip_file(path: Path) -> bool:
    skip_names = {".DS_Store", ".gitkeep"}
    return path.name in skip_names or "__MACOSX" in path.parts


def _resolve_local_path(path_value: str | Path) -> Path:
    path_value = Path(path_value).expanduser()

    if path_value.is_absolute():
        return path_value.resolve()

    cwd_candidate = (Path.cwd() / path_value).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = (PROJECT_ROOT / path_value).resolve()
    if project_candidate.exists():
        return project_candidate

    return project_candidate


class S3DataAccess:
    def __init__(self, allow_http: str = "False") -> None:
        self._endpoint = os.environ.get("DIH_S3_ENDPOINT", "test_default")
        self._access_key = os.environ.get("DIH_S3_TOKEN", "test_default")
        self._secret_key = os.environ.get("DIH_S3_SECRET", "not_required")
        self._region = os.environ.get("DIH_S3_REGION", "eu-central-1")
        self._bucket = os.environ.get("DIH_S3_BUCKET_NAME", "ka-etu-dih-001-oktoplustim-001")
        self._verify_ssl = os.environ.get("DIH_S3_VERIFY_SSL", "True").lower() == "true"

        self._STORAGE_OPTIONS = {
            "ENDPOINT_URL": self._endpoint,
            "AWS_ACCESS_KEY_ID": self._access_key,
            "AWS_SECRET_ACCESS_KEY": self._secret_key,
            "AWS_REGION": self._region,
            "aws_allow_http": allow_http,
        }

        self._s3_client = None
        self._overwrite_mode = "ask"

    def read_data(self, path: str) -> pl.LazyFrame:
        table = DeltaTable(path, storage_options=self._STORAGE_OPTIONS)
        lf = pl.LazyFrame(table.to_pyarrow_table())
        return _convert_datetimes_to_timezone(lf, TIMEZONE)

    def _get_s3_client(self):
        if self._s3_client is None:
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self._endpoint,
                aws_access_key_id=self._access_key,
                aws_secret_access_key=self._secret_key,
                region_name=self._region,
                verify=self._verify_ssl,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                ),
            )
        return self._s3_client

    def _set_overwrite_mode(self, overwrite_mode: str) -> None:
        if overwrite_mode not in {"ask", "all", "none"}:
            raise ValueError(f"Ungültiger overwrite_mode: {overwrite_mode}")
        self._overwrite_mode = overwrite_mode

    def _object_exists(self, target_key: str) -> bool:
        try:
            self._get_s3_client().head_object(Bucket=self._bucket, Key=target_key)
            return True
        except ClientError as exc:
            error = exc.response.get("Error", {})
            code = str(error.get("Code", ""))
            status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

            if code in {"404", "NoSuchKey", "NotFound"} or status_code == 404:
                return False
            raise

    def _prompt_overwrite(self, target_key: str) -> str:
        prompt = (
            f"\nDatei existiert bereits:\n"
            f"s3://{self._bucket}/{target_key}\n"
            f"Ersetzen? [j]a / [n]ein / [a]lle ja / [k]eine mehr: "
        )

        mapping = {
            "j": "yes",
            "ja": "yes",
            "y": "yes",
            "n": "no",
            "nein": "no",
            "a": "all",
            "alle": "all",
            "all": "all",
            "k": "none",
            "keine": "none",
            "none": "none",
        }

        while True:
            answer = input(prompt).strip().lower()
            if answer in mapping:
                return mapping[answer]
            print("Bitte nur j, n, a oder k eingeben.")

    def _should_upload_target(self, target_key: str) -> bool:
        if not self._object_exists(target_key):
            return True

        if self._overwrite_mode == "all":
            return True

        if self._overwrite_mode == "none":
            return False

        decision = self._prompt_overwrite(target_key)

        if decision == "yes":
            return True

        if decision == "no":
            return False

        if decision == "all":
            self._overwrite_mode = "all"
            return True

        if decision == "none":
            self._overwrite_mode = "none"
            return False

        return False

    def _upload_directory_impl(
            self,
            local_dir: Path,
            target_prefix: str,
            dry_run: bool = True,
    ) -> dict[str, list[tuple[str, str]]]:
        if not local_dir.exists():
            raise FileNotFoundError(f"Ordner nicht gefunden: {local_dir}")
        if not local_dir.is_dir():
            raise NotADirectoryError(f"Kein Ordner: {local_dir}")

        normalized_prefix = target_prefix.strip("/")
        summary = {
            "uploaded": [],
            "skipped": [],
        }

        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            if _should_skip_file(path):
                continue

            relative_path = path.relative_to(local_dir).as_posix()
            target_key = f"{normalized_prefix}/{relative_path}" if normalized_prefix else relative_path

            if dry_run:
                summary["uploaded"].append((str(path), target_key))
                continue

            if not self._should_upload_target(target_key):
                summary["skipped"].append((str(path), target_key))
                continue

            self._get_s3_client().upload_file(str(path), self._bucket, target_key)
            summary["uploaded"].append((str(path), target_key))

        return summary

    def upload_checkpoints(
            self,
            dry_run: bool = True,
            target_prefix: str = "data/tab/checkpoints/",
            overwrite_mode: str = "ask",
    ) -> dict[str, list[tuple[str, str]]]:
        llm_dir = PROJECT_ROOT / "ts_benchmark" / "baselines" / "LLM" / "checkpoints"
        pre_train_dir = PROJECT_ROOT / "ts_benchmark" / "baselines" / "pre_train" / "checkpoints"

        self._set_overwrite_mode(overwrite_mode)

        summary = {
            "uploaded": [],
            "skipped": [],
        }

        llm_summary = self._upload_directory_impl(
            local_dir=llm_dir,
            target_prefix=f"{target_prefix.strip('/')}/LLM/checkpoints/",
            dry_run=dry_run,
        )
        summary["uploaded"].extend(llm_summary["uploaded"])
        summary["skipped"].extend(llm_summary["skipped"])

        pre_train_summary = self._upload_directory_impl(
            local_dir=pre_train_dir,
            target_prefix=f"{target_prefix.strip('/')}/pre_train/checkpoints/",
            dry_run=dry_run,
        )
        summary["uploaded"].extend(pre_train_summary["uploaded"])
        summary["skipped"].extend(pre_train_summary["skipped"])

        return summary

    def upload_file(
            self,
            local_file: str | Path,
            target_key: str,
            dry_run: bool = True,
            overwrite_mode: str = "ask",
    ) -> tuple[str, str] | None:
        self._set_overwrite_mode(overwrite_mode)

        local_file = _resolve_local_path(local_file)
        if not local_file.is_file():
            raise FileNotFoundError(f"Datei nicht gefunden: {local_file}")

        normalized_key = target_key.strip("/")

        if dry_run:
            return str(local_file), normalized_key

        if not self._should_upload_target(normalized_key):
            return None

        self._get_s3_client().upload_file(str(local_file), self._bucket, normalized_key)
        return str(local_file), normalized_key

    def upload_directory(
            self,
            local_dir: str | Path,
            target_prefix: str = "data/tab/dataset/",
            dry_run: bool = True,
            overwrite_mode: str = "ask",
    ) -> dict[str, list[tuple[str, str]]]:
        self._set_overwrite_mode(overwrite_mode)
        local_dir = _resolve_local_path(local_dir)
        return self._upload_directory_impl(
            local_dir=local_dir,
            target_prefix=target_prefix,
            dry_run=dry_run,
        )


def main() -> int:
    if load_dotenv is not None:
        load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="DIH S3 / Delta Test- und Upload-Skript")
    parser.add_argument("--read-test", action="store_true", help="Delta-Tabelle lesen und testen")
    parser.add_argument(
        "--upload-test",
        metavar="LOCAL_DIR",
        help="Dry-Run für den Upload eines lokalen Ordners nach data/tab/dataset/",
    )
    parser.add_argument(
        "--upload",
        metavar="LOCAL_DIR",
        help="Echten Upload eines lokalen Ordners nach data/tab/dataset/ ausführen",
    )
    parser.add_argument(
        "--dataset-prefix",
        default=os.environ.get("DIH_S3_DATASET_PREFIX", "data/tab/dataset/"),
        help="Zielprefix für rohe Dateien, Standard: data/tab/dataset/",
    )
    parser.add_argument(
        "--table-path",
        default=os.environ.get(
            "DIH_S3_TABLE_PATH",
            f"s3://{os.environ.get('DIH_S3_BUCKET_NAME', 'ka-etu-dih-001-oktoplustim-001')}/data/tab/table/",
        ),
        help="Pfad zur Delta-Tabelle",
    )
    parser.add_argument(
        "--upload-checkpoints-test",
        action="store_true",
        help="Dry-Run für den Upload der Checkpoints nach data/tab/checkpoints/",
    )
    parser.add_argument(
        "--upload-checkpoints",
        action="store_true",
        help="Echten Upload der Checkpoints nach data/tab/checkpoints/ ausführen",
    )
    parser.add_argument(
        "--checkpoints-prefix",
        default=os.environ.get("DIH_S3_CHECKPOINTS_PREFIX", "data/tab/checkpoints/"),
        help="Zielprefix für Checkpoints, Standard: data/tab/checkpoints/",
    )
    parser.add_argument(
        "--overwrite",
        choices=["ask", "all", "none"],
        default=os.environ.get("DIH_S3_OVERWRITE_MODE", "ask"),
        help="Verhalten bei bereits existierenden Dateien: ask, all oder none",
    )
    args = parser.parse_args()

    endpoint = os.environ.get("DIH_S3_ENDPOINT")
    token = os.environ.get("DIH_S3_TOKEN")
    allow_http = os.environ.get("DIH_S3_ALLOW_HTTP", "False")
    bucket = os.environ.get("DIH_S3_BUCKET_NAME", "ka-etu-dih-001-oktoplustim-001")

    if not endpoint:
        print("Fehler: DIH_S3_ENDPOINT ist nicht gesetzt.")
        return 1

    if not token:
        print("Fehler: DIH_S3_TOKEN ist nicht gesetzt.")
        return 1

    access = S3DataAccess(allow_http=allow_http)

    if args.upload_test:
        resolved_dir = _resolve_local_path(args.upload_test)
        print("Starte Upload-Dry-Run...")
        print(f"Endpoint: {endpoint}")
        print(f"Bucket: {bucket}")
        print(f"Lokaler Ordner: {resolved_dir}")
        print(f"Zielprefix: {args.dataset_prefix}")
        print("Hinweis: Im Dry-Run wird nicht geprüft, ob Dateien im Bucket bereits existieren.")

        try:
            summary = access.upload_directory(
                local_dir=args.upload_test,
                target_prefix=args.dataset_prefix,
                dry_run=True,
                overwrite_mode=args.overwrite,
            )
            print(f"\nGeplante Uploads: {len(summary['uploaded'])}")
            for source_path, target_key in summary["uploaded"][:20]:
                print(f"{source_path} -> s3://{bucket}/{target_key}")
            if len(summary["uploaded"]) > 20:
                print(f"... und {len(summary['uploaded']) - 20} weitere Dateien")
            return 0
        except Exception as exc:
            print("\nUpload-Dry-Run fehlgeschlagen.")
            print(f"{type(exc).__name__}: {exc}")
            return 2

    if args.upload_checkpoints_test:
        print("Starte Checkpoints-Upload-Dry-Run...")
        print(f"Endpoint: {endpoint}")
        print(f"Bucket: {bucket}")
        print(f"Zielprefix: {args.checkpoints_prefix}")
        print("Hinweis: Im Dry-Run wird nicht geprüft, ob Dateien im Bucket bereits existieren.")

        try:
            summary = access.upload_checkpoints(
                dry_run=True,
                target_prefix=args.checkpoints_prefix,
                overwrite_mode=args.overwrite,
            )
            print(f"\nGeplante Uploads: {len(summary['uploaded'])}")
            for source_path, target_key in summary["uploaded"][:20]:
                print(f"{source_path} -> s3://{bucket}/{target_key}")
            if len(summary["uploaded"]) > 20:
                print(f"... und {len(summary['uploaded']) - 20} weitere Dateien")
            return 0
        except Exception as exc:
            print("\nCheckpoints-Upload-Dry-Run fehlgeschlagen.")
            print(f"{type(exc).__name__}: {exc}")
            return 2

    if args.upload_checkpoints:
        print("Starte echten Checkpoints-Upload...")
        print(f"Endpoint: {endpoint}")
        print(f"Bucket: {bucket}")
        print(f"Zielprefix: {args.checkpoints_prefix}")
        print(f"Overwrite-Modus: {args.overwrite}")

        try:
            summary = access.upload_checkpoints(
                dry_run=False,
                target_prefix=args.checkpoints_prefix,
                overwrite_mode=args.overwrite,
            )
            print(f"\nUpload erfolgreich. Hochgeladen: {len(summary['uploaded'])}")
            print(f"Übersprungen: {len(summary['skipped'])}")
            return 0
        except Exception as exc:
            print("\nCheckpoints-Upload fehlgeschlagen.")
            print(f"{type(exc).__name__}: {exc}")
            return 2

    if args.upload:
        resolved_dir = _resolve_local_path(args.upload)
        print("Starte echten Upload...")
        print(f"Endpoint: {endpoint}")
        print(f"Bucket: {bucket}")
        print(f"Lokaler Ordner: {resolved_dir}")
        print(f"Zielprefix: {args.dataset_prefix}")
        print(f"Overwrite-Modus: {args.overwrite}")

        try:
            summary = access.upload_directory(
                local_dir=args.upload,
                target_prefix=args.dataset_prefix,
                dry_run=False,
                overwrite_mode=args.overwrite,
            )
            print(f"\nUpload erfolgreich. Hochgeladen: {len(summary['uploaded'])}")
            print(f"Übersprungen: {len(summary['skipped'])}")
            return 0
        except Exception as exc:
            print("\nUpload fehlgeschlagen.")
            print(f"{type(exc).__name__}: {exc}")
            return 2

    print("Starte Delta-Lese-Test...")
    print(f"Endpoint: {endpoint}")
    print(f"Pfad: {args.table_path}")

    try:
        lf = access.read_data(args.table_path)

        print("\nSchema:")
        print(lf.collect_schema())

        row_count = lf.select(pl.len().alias("row_count")).collect()
        print("\nZeilenanzahl:")
        print(row_count)

        print("\nErste 5 Zeilen:")
        print(lf.limit(5).collect())

        print("\nTest erfolgreich.")
        return 0
    except Exception as exc:
        print("\nTest fehlgeschlagen.")
        print(f"{type(exc).__name__}: {exc}")
        print("\nHinweis: Der Pfad muss auf eine echte Delta-Tabelle zeigen (Ordner mit _delta_log).")
        print("Für Rohdaten verwenden Sie den Upload nach data/tab/dataset/.")
        return 2


if __name__ == "__main__":
    sys.exit(main())