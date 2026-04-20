import { Component, Input } from '@angular/core';
import {TitleCasePipe} from "@angular/common";
import {TranslatePipe} from "@ngx-translate/core";
import { DatasetToReviewResponseDto } from '../../../../core/api/models/dataset-to-review-response-dto';
import { DatasetStatus } from '../../../../core/api/models/dataset-status';
import { DatasetStatusModel } from '../../../../models/dataset-status.model';
import { ReviewDatasetItem } from '../review-dataset-item/review-dataset-item';

@Component({
  selector: 'app-dataset-item',
  imports: [TitleCasePipe, TranslatePipe, ReviewDatasetItem],
  templateUrl: './dataset-item.html',
  styleUrl: './dataset-item.css',
})
export class DatasetItem {
  @Input({ required: true }) dataset!: DatasetToReviewResponseDto;
  @Input() index!: number;

  isDatasetReviewModalOpen = false;

  public getDatasetStatusName(status: DatasetStatus): string {
    return DatasetStatusModel[status];
  }

  public openReviewDatasetItem(): void {
    this.isDatasetReviewModalOpen = true;
  }

  public closeReviewDatasetItem(): void {
    this.isDatasetReviewModalOpen = false;
  }
}
