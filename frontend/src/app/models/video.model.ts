import { Dataset } from '../core/api/models/dataset';
import { DatasetModel } from './dataset.model';

export interface Video {
  id: number;
  name: string;
  path: string;
  is_violent: boolean;
  duration: number;
  dataset: DatasetModel;
}
