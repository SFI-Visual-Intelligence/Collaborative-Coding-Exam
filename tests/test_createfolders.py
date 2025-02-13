from CollaborativeCoding import createfolders


def test_createfolders():
    import argparse
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        parser = argparse.ArgumentParser()

        # Structuture related values
        parser.add_argument(
            "--datafolder",
            type=Path,
            default=temp_dir / "Data",
            help="Path to where data will be saved during training.",
        )
        parser.add_argument(
            "--resultfolder",
            type=Path,
            default=temp_dir / "Results",
            help="Path to where results will be saved during evaluation.",
        )
        parser.add_argument(
            "--modelfolder",
            type=Path,
            default=temp_dir / "Experiments",
            help="Path to where model weights will be saved at the end of training.",
        )

        args = parser.parse_args(
            [
                "--datafolder",
                str(temp_dir / "Data"),
                "--resultfolder",
                str(temp_dir / "Results"),
                "--modelfolder",
                str(temp_dir / "Experiments"),
            ]
        )

        createfolders(args.datafolder, args.resultfolder, args.modelfolder)

        assert (temp_dir / "Data").exists()
        assert (temp_dir / "Results").exists()
        assert (temp_dir / "Experiments").exists()
