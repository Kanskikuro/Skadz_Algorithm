import scripts.download_champions_and_icons as download_champions_and_icons
import scripts.download_dataset as download_dataset
import scripts.process_dataset as process_dataset
if __name__ == "__main__":
    download_champions_and_icons.main()
    download_dataset.main()
    process_dataset.main()