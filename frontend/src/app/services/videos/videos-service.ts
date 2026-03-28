import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { VideoResponseDto } from '../../core/api/models/video-response-dto';
import { environment } from '../../../environments/environment.development';

@Injectable({
  providedIn: 'root',
})
export class VideosService {
  constructor(private httpClient: HttpClient) {}

  public getVideosPaged(
    asc: boolean,
    page: number,
    page_size: number,
    search_term?: string,
    dataset_id?: number,
  ): Observable<VideoResponseDto[]> {
    let params = new HttpParams().set('page', page).set('page_size', page_size).set('asc', asc);

    if (search_term) {
      params = params.set('search_term', search_term);
    }
    if (dataset_id && dataset_id !== 0) {
      params = params.set('dataset_id', dataset_id);
    }
    return this.httpClient.get<VideoResponseDto[]>(
      `${environment.apiUrl}videos/get_videos_paged`,
      {
        params: params,
        withCredentials: true,
      },
    );
  }
}
