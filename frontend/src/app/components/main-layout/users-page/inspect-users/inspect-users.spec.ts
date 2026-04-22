import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InspectUsers } from './inspect-users';

describe('InspectUsers', () => {
  let component: InspectUsers;
  let fixture: ComponentFixture<InspectUsers>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InspectUsers],
    }).compileComponents();

    fixture = TestBed.createComponent(InspectUsers);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
