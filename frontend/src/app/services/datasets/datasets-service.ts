import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { DatasetResponseDto } from '../../core/api/models/dataset-response-dto';
import { environment } from '../../../environments/environment.development';
import { DatasetStatus } from '../../core/api/models/dataset-status';
import { DatasetToReviewResponseDto } from '../../core/api/models/dataset-to-review-response-dto';
import { DatasetWithVideosResponseDto } from '../../core/api/models/dataset-with-videos-response-dto';

@Injectable({
  providedIn: 'root',
})
export class DatasetsService {
  constructor(private httpClient: HttpClient) {}

  public getAcceptedDatasets(): Observable<DatasetResponseDto[]> {
    return this.httpClient.get<DatasetResponseDto[]>(
      environment.apiUrl + 'datasets/get_accepted_datasets',
      {
        withCredentials: true,
      },
    );
  }

  public createUnofficialDataset(formData: FormData): Observable<void> {
    return this.httpClient.post<void>(
      `${environment.apiUrl}datasets/create_unofficial_dataset`,
      formData,
      {
        withCredentials: true,
      },
    );
  }

  public getDatasets(
    searchTerm: string,
    page: number,
    pageSize: number,
    status: DatasetStatus | null,
  ): Observable<DatasetToReviewResponseDto[]> {
    let params: HttpParams = new HttpParams()
      .set('page', page)
      .set('page_size', pageSize);
    if (searchTerm !== '') {
      params = params.set('search_term', searchTerm);
    }
    if (status !== null) {
      params = params.set('dataset_status', status);
    }
    return this.httpClient.get<DatasetToReviewResponseDto[]>(
      environment.apiUrl + 'datasets/get_datasets',
      {
        params: params,
        withCredentials: true,
      },
    );
  }

  public getDatasetWithVideos(datasetId: number): Observable<DatasetWithVideosResponseDto> {
    return this.httpClient.get<DatasetWithVideosResponseDto>(
      environment.apiUrl + `datasets/get_dataset_videos/${datasetId}`, { withCredentials: true },
    );
  }

  public reviewDataset(datasetId: number, isApproved: boolean, videos: { video_id: number, is_violent: boolean }[], reviewComment: string): Observable<DatasetResponseDto> {
    const body = {
      is_approved: isApproved,
      videos: videos,
      review_comment: reviewComment,
    };
    return this.httpClient.patch<DatasetResponseDto>(
      `${environment.apiUrl}datasets/review_dataset/${datasetId}`,
      body,
      { withCredentials: true },
    );
  }
}
