import { Component, EventEmitter, Input, Output } from '@angular/core';
import {TitleCasePipe} from "@angular/common";
import {TranslatePipe} from "@ngx-translate/core";
import { DatasetToReviewResponseDto } from '../../../../core/api/models/dataset-to-review-response-dto';
import { DatasetStatus } from '../../../../core/api/models/dataset-status';
import { DatasetStatusModel } from '../../../../models/dataset-status.model';
import { ReviewDatasetItem } from '../review-dataset-item/review-dataset-item';
import { ConfirmationPopup } from '../../../confirmation-popup/confirmation-popup';
import { EditDatasetItem } from '../edit-dataset-item/edit-dataset-item';

@Component({
  selector: 'app-dataset-item',
  imports: [TitleCasePipe, TranslatePipe, ReviewDatasetItem, ConfirmationPopup, EditDatasetItem],
  templateUrl: './dataset-item.html',
  styleUrl: './dataset-item.css',
})
export class DatasetItem {
  @Input({ required: true }) dataset!: DatasetToReviewResponseDto;
  @Input() index!: number;
  @Output() deleteDatasetItem: EventEmitter<void> = new EventEmitter();

  isDatasetReviewModalOpen: boolean = false;
  isDeleteConfirmationModalOpen: boolean = false;
  isEditDatasetItemModalOpen: boolean = false;

  public getDatasetStatusName(status: DatasetStatus): string {
    return DatasetStatusModel[status];
  }

  public openReviewDatasetItem(): void {
    if(this.dataset.status !== DatasetStatusModel.PENDING)
      return;
    this.isDatasetReviewModalOpen = true;
  }

  public closeReviewDatasetItem(): void {
    this.isDatasetReviewModalOpen = false;
  }

  protected readonly DatasetStatusModel = DatasetStatusModel;

  public openDeleteDatasetConfirmationModal() {
    this.isDeleteConfirmationModalOpen = true;
  }

  public deleteDataset(): void {
    this.deleteDatasetItem.emit();
  }

  public closeDeleteDatasetConfirmationModal(): void {
    this.isDeleteConfirmationModalOpen = false;
  }

  public openEditDatasetItem() {
    this.isEditDatasetItemModalOpen = true;
  }

  public closeEditDatasetItem() {
    this.isEditDatasetItemModalOpen = false;
  }
}
