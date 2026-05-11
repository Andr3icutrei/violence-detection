import { ChangeDetectorRef, Component, Input, OnInit } from '@angular/core';
import { SearchBar } from '../../../../search-bar/search-bar';
import { DatasetsService } from '../../../../../services/datasets/datasets-service';
import { TranslatePipe } from '@ngx-translate/core';
import { Paginator } from '../../../../paginator/paginator';
import { DatasetToReviewResponseDto } from '../../../../../core/api/models/dataset-to-review-response-dto';

@Component({
  selector: 'app-datasets-panel',
  imports: [SearchBar, TranslatePipe, Paginator],
  templateUrl: './datasets-panel.html',
  styleUrl: './datasets-panel.css',
})
export class DatasetsPanel implements OnInit {
  @Input({ required: true }) isOfficial!: boolean;
  searchTerm: string = '';
  readonly pageSize: number = 10;
  page: number = 0;
  hasMore: boolean = false;

  availableDatasets: DatasetToReviewResponseDto[] = [];

  constructor(
    private cdr: ChangeDetectorRef,
    private datasetsService: DatasetsService,
  ) {}

  ngOnInit(): void {
    this.loadDatasets();
  }

  public debouncedSearch(searchTerm: string) {
    this.searchTerm = searchTerm;
    this.page = 0;
    this.loadDatasets();
  }

  public loadDatasets(): void {
    this.datasetsService
      .getDatasets(this.searchTerm, this.page, this.pageSize, 20, this.isOfficial)
      .subscribe((datasets) => {
        this.availableDatasets = datasets;
        this.hasMore = datasets.length > this.pageSize;
        this.cdr.detectChanges();
      });
  }

  public onPageChange(pageNumber: number) {
    this.page = pageNumber;
    this.loadDatasets();
  }
}
