import { ChangeDetectorRef, Component, OnDestroy, OnInit } from '@angular/core';
import { DatasetsService } from '../../../../services/datasets/datasets-service';
import { DatasetStatus } from '../../../../core/api/models/dataset-status';
import { DatasetToReviewResponseDto } from '../../../../core/api/models/dataset-to-review-response-dto';
import { TranslatePipe } from '@ngx-translate/core';
import {DatasetStatusModel} from '../../../../models/dataset-status.model';
import { TitleCasePipe} from '@angular/common';
import {DatasetItem} from '../dataset-item/dataset-item';
import {SearchBar} from '../../../search-bar/search-bar';
import { Paginator } from '../../../paginator/paginator';
import { ReactiveFormsModule } from '@angular/forms';
import { HttpErrorResponse } from '@angular/common/http';
import { DatasetUpdatedService } from '../../../../services/dataset_updated/dataset-updated.service';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-inspect-datasets',
  imports: [TranslatePipe, DatasetItem, SearchBar, Paginator, ReactiveFormsModule, TitleCasePipe],
  templateUrl: './inspect-datasets.html',
  styleUrl: './inspect-datasets.css',
})
export class InspectDatasets implements OnInit {
  readonly pageSize: number = 10;
  searchTerm: string = '';
  page: number = 0;
  selectedDatasetStatus: DatasetStatus | null = null;
  isLoadingDatasets: boolean = false;
  hasMoreDatasets: boolean = false;

  datasetsToReview: DatasetToReviewResponseDto[] = [];

  statusOptions: DatasetStatusModel[] = Object.values(DatasetStatusModel).filter(
    (value) => typeof value === 'number',
  ) as DatasetStatusModel[];

  constructor(
    private cdr: ChangeDetectorRef,
    private datasetsService: DatasetsService,
  ) {}

  ngOnInit(): void {
    this.loadDatasets();
  }

  public debouncedSearch(searchTerm: string): void {
    this.searchTerm = searchTerm;
    this.loadDatasets();
  }

  public onPageChange(pageNumber: number): void {
    this.page = pageNumber;
  }

  public getDatasetStatusName(status: DatasetStatus): string {
    return DatasetStatusModel[status];
  }

  public onSelectedDatasetStatusChange(event: Event): void {
    const target = event.target as HTMLSelectElement | null;
    if (!target) {
      return;
    }
    const value = target.value;
    if (!value || value === 'null') {
      this.selectedDatasetStatus = null;
    } else {
      this.selectedDatasetStatus = Number(value) as unknown as DatasetStatus;
    }
    this.searchTerm = '';
    this.loadDatasets();
  }

  private loadDatasets(): void {
    this.datasetsToReview = [];
    this.isLoadingDatasets = true;
    this.datasetsService
      .getDatasets(this.searchTerm, this.page, this.pageSize, this.selectedDatasetStatus)
      .subscribe({
        next: (data: DatasetToReviewResponseDto[]): void => {
          this.hasMoreDatasets = data.length > this.pageSize;
          this.datasetsToReview = data;
          this.isLoadingDatasets = false;
          this.cdr.detectChanges();
        },
        error: (error) => {
          this.isLoadingDatasets = false;
          this.cdr.detectChanges();
        },
      });
  }

  public deleteDataset(datasetId: number) {
    this.datasetsService.deleteDataset(datasetId).subscribe({
      next: () => {
        this.resetForm();
        this.loadDatasets();
        this.cdr.detectChanges();
      },
      error: (error: HttpErrorResponse) => {
        console.log(error);
      },
    });
  }

  public resetForm(): void {
    this.searchTerm = '';
    this.page = 0;
    this.selectedDatasetStatus = null;
    this.isLoadingDatasets = false;
    this.hasMoreDatasets = false;
  }
}
