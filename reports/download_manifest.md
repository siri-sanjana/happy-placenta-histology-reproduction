# Public download manifest

## Official sources and expected files

| Resource | Official source | Expected size | Published checksum / license | Workspace status |
|---|---|---:|---|---|
| HAPPY archive 001 | [Zenodo 10535021](https://zenodo.org/records/10535021) | 2,141,793,908 bytes | MD5 `e8af7d9a349051e209c9d982bc44edcf`; CC BY 4.0 | **Verified** size, MD5 and zip CRC; payload extracted into `data/prepared/` |
| HAPPY archive 002 | [Zenodo 10535021](https://zenodo.org/records/10535021) | 452,945,061 bytes | MD5 `e39adcab4ea865f8d498d62ced6b1ca6`; CC BY 4.0 | **Verified** size, MD5, and `unzip -t`; extracted copy at `data/verified_archive_002/HAPPYData` contains 8,079 cell-classification patch PNG files |
| Graph slide `96-2019-09-05 22.47.40.ndpi` | [BioImage Archive S-BIAD1045](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1045) | 2,686,927,912 bytes | CC BY 4.0 | **Size verified**; TIFF signature recognized; no public checksum listed |
| Graph slide `59B.svs` | [BioImage Archive S-BIAD1045](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD1045) | 2,929,818,169 bytes | CC BY 4.0 | In progress (2,395,348,992 bytes at last check; resumable partial retained; existing curl session is retrying after a connection reset) |

Both Zenodo archives are verified against expected byte counts and MD5 values, and pass `unzip -t`. The preparation script safely merged their shared `HAPPYData/` root into `data/prepared/`; all six expected top-level folders are present. The extracted data includes 20 annotation files, 19,221 dataset patch files, two HDF5 embedding files (`lab_1/wsi_1/run_1.hdf5`, `lab_2/wsi_2/run_2.hdf5`), the original graph split definitions, one released demo slide, and three released weights. Both HDF5 files contain aligned `predictions`, 64-D `embeddings`, `coords`, and `confidence` datasets: 1,307,090 rows for WSI 1 and 1,238,909 for WSI 2; matching graph annotation TSVs have those row counts. Cell patches in archive 002 alone are not the complete cell dataset; the merged release contains the full annotation/split files required by the paper-era runner. The public BIA raw WSI files are supplementary raw microscopy slides; training uses the released patches/embeddings and does not need to reprocess those WSIs. The first BIA slide is complete and size-verified; the second is still transferring. Earlier Zenodo partials were range-resumed; the verified archive files were not overwritten.

Expected raw public download size: 2,594,728,969 bytes for both Zenodo archives plus 5,616,746,081 bytes for both BioImage Archive WSIs, 8,211,475,050 bytes total (about 8.21 GB decimal). This excludes supplementary paper files and extracted contents.

## Paper files

| File | Source | Status |
|---|---|---|
| `references/paper.pdf` | PMC/Nature article DOI 10.1038/s41467-024-46986-2 | Downloaded; PDF signature checked |
| `references/paper_fulltext.xml` | Europe PMC open full-text endpoint | Downloaded |
| `references/supplement_1.pdf` | Nature supplementary information | Downloaded |
| `references/supplement_2.pdf` | Nature reporting summary | Downloaded |
| `references/supplement_3.pdf` | Nature peer review file | Downloaded |
| `references/source_data.xlsx` | Nature source data | Downloaded |

## Verification commands

After transfers finish:

```sh
stat -f '%N %z bytes' data/raw_public/HAPPYData-20240119T100520Z-*.zip
md5 data/raw_public/HAPPYData-20240119T100520Z-001.zip
md5 data/raw_public/HAPPYData-20240119T100520Z-002.zip
unzip -t data/raw_public/HAPPYData-20240119T100520Z-001.zip
unzip -t data/raw_public/HAPPYData-20240119T100520Z-002.zip
```
