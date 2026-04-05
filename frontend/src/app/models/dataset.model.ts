import { UserResponseDto } from '../core/api/models/user-response-dto';

export interface DatasetModel {
  id: number;
  name: string;
  is_official: boolean;
  user?: UserResponseDto;
}
